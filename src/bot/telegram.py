"""The Telegram bot: chat with the video library from Telegram.

    python -m src.bot.telegram        # compose runs it as the `telegram-bot` service

A thin client, like the Airflow DAGs: every message goes to POST /api/v1/chat; the agent lives in the API.
- Long polling (getUpdates), so no public URL is needed; on AWS this can become a webhook.
- One conversation per Telegram chat; /new starts a fresh one.
- Clips are sent as video files read from S3: Telegram's servers can't open our presigned localhost URLs.
- After a Pexels download it keeps polling /chat/{id}/updates and sends the clip when it's ready.
The bot token is a secret: it's in every Bot API URL, so HTTP request logging is turned off.
"""

import logging
import threading
import time
import uuid
from typing import Any

import httpx

from src.config import Settings, check_production
from src.services.identity import SERVICE_TOKEN_HEADER, USER_HEADER
from src.services.storage import StorageClient, make_storage_client

logger = logging.getLogger("telegram-bot")

HELP = (
    "Hi! I find the exact moment you're looking for in a video and send it back as a short clip.\n\n"
    "🎥 Got a video? Send it to me (up to 20 MB and 10 minutes), then tell me which part you want, for example:\n"
    "“the part where she says thank you”\n"
    "No need to wait: ask right away, and I'll send the clip as soon as your video is ready (usually a minute or two).\n\n"
    "🔎 Need some footage? Just describe it, like “a dog playing in the snow”. If I don't have it, I can look for free "
    "stock videos.\n\n"
    "❓ Have a question, like “how long is a sleep cycle?”? I'll look through the videos in my library for one that "
    "answers it and send you that clip. If none of them does, I'll tell you.\n\n"
    "🖼 Send me a photo and I'll find the scene that looks most like it.\n\n"
    "🔒 Videos you send are private to you and deleted after 7 days. Type /delete to remove yours sooner.\n\n"
    "⚖️ To keep this free for everyone, each person gets a daily allowance of AI tokens: plenty for about a hundred "
    "requests or a few videos. It resets every day, and /usage shows how much you have left.\n\n"
    "Type /new to start over."
)
MAX_UPLOAD_BYTES = 45 * 1024 * 1024  # Telegram bots may upload up to 50 MB
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # …and download at most 20 MB (getFile)


class RedactToken(logging.Filter):
    """Errors can quote a Bot API URL, and the URL contains the token: never let it reach the logs."""

    def __init__(self, token: str):
        super().__init__()
        self.token = token

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + logging.Formatter().formatException(record.exc_info)
            record.exc_info, record.exc_text = None, None
        record.msg, record.args = message.replace(self.token, "<token>"), ()
        return True


class TelegramBot:
    def __init__(
        self,
        token: str,
        api_url: str,
        storage: StorageClient,
        allowed_chat_ids: set[int] | None = None,
        service_token: str = "",
    ):
        self.telegram = httpx.Client(base_url=f"https://api.telegram.org/bot{token}", timeout=70)
        # The service token lets the API trust the user id we send: each Telegram user owns their uploads.
        self.api = httpx.Client(base_url=api_url, timeout=300, headers={SERVICE_TOKEN_HEADER: service_token})
        self.storage = storage
        self.allowed = allowed_chat_ids or set()
        self.conversations: dict[int, str] = {}  # Telegram chat → our conversation id
        self.watching: set[str] = set()  # conversations with a status message being kept up to date

    # ---- Telegram Bot API ----------------------------------------------------------------------------------------
    def call(self, method: str, **params: Any) -> Any:
        files = params.pop("files", None)
        response = self.telegram.post(f"/{method}", data=params if files else None, json=None if files else params, files=files)
        body = response.json()
        if not body.get("ok"):
            raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
        return body["result"]

    def send_text(self, chat_id: int, text: str) -> int | None:
        return (self.call("sendMessage", chat_id=chat_id, text=text[:4096]) or {}).get("message_id")

    def edit_text(self, chat_id: int, message_id: int | None, text: str) -> None:
        if message_id is None:
            return
        try:
            self.call("editMessageText", chat_id=chat_id, message_id=message_id, text=text[:4096])
        except RuntimeError:  # "message is not modified", or deleted by the user: not worth failing over
            pass

    def send_clip(self, chat_id: int, clip: dict[str, Any]) -> None:
        caption = (clip.get("explanation") or "")[:1024]
        data = self.storage.get_bytes(clip["key"]) if clip.get("key") else None
        if data is None or len(data) > MAX_UPLOAD_BYTES:
            self.send_text(chat_id, f"{caption}\n{clip['url']}")
            return
        self.call("sendVideo", chat_id=chat_id, caption=caption, supports_streaming="true",
                  files={"video": ("clip.mp4", data, "video/mp4")})  # fmt: skip

    def file_bytes(self, file_id: str) -> bytes:
        path = self.call("getFile", file_id=file_id)["file_path"]
        base = str(self.telegram.base_url).replace("/bot", "/file/bot", 1)
        return httpx.get(f"{base}/{path}", timeout=300).raise_for_status().content

    def photo_bytes(self, photo_sizes: list[dict[str, Any]]) -> bytes:
        return self.file_bytes(max(photo_sizes, key=lambda p: p.get("file_size", 0))["file_id"])

    @staticmethod
    def attached_video(message: dict[str, Any]) -> dict[str, Any] | None:
        """A video sent as a video, or as a file ("document") with a video type."""
        if message.get("video"):
            return message["video"]
        document = message.get("document") or {}
        return document if (document.get("mime_type") or "").startswith("video/") else None

    # ---- one message -----------------------------------------------------------------------------------------------
    def conversation(self, chat_id: int) -> str:
        return self.conversations.setdefault(chat_id, f"tg-{chat_id}-{uuid.uuid4().hex[:8]}")

    @staticmethod
    def user(message: dict[str, Any]) -> dict[str, str]:
        """Who sent it, as the API's viewer id: their uploads are private to them."""
        return {USER_HEADER: f"tg:{(message.get('from') or {}).get('id') or message['chat']['id']}"}

    def handle(self, message: dict[str, Any]) -> None:
        chat_id = message["chat"]["id"]
        if self.allowed and chat_id not in self.allowed:
            self.send_text(chat_id, "Sorry, this bot is private.")
            return
        text = (message.get("text") or message.get("caption") or "").strip()
        if text in ("/start", "/help"):
            self.send_text(chat_id, HELP)
            return
        if text == "/usage":
            self.send_text(chat_id, self.usage_text(message))
            return
        if text == "/delete":
            text = "delete my video"
        if text == "/new":
            self.conversations.pop(chat_id, None)
            self.send_text(chat_id, "New conversation. What would you like to see?")
            return

        files = None
        video = self.attached_video(message)
        if video is not None:
            if (video.get("file_size") or 0) > MAX_DOWNLOAD_BYTES:
                self.send_text(chat_id, "That video is over 20 MB, the most a Telegram bot can download. "
                               "Send a shorter clip, or use the web page.")  # fmt: skip
                return
            self.call("sendChatAction", chat_id=chat_id, action="upload_document")
            name = video.get("file_name") or "video.mp4"
            files = {"video": (name, self.file_bytes(video["file_id"]), video.get("mime_type") or "video/mp4")}
        elif message.get("photo"):
            files = {"image": ("photo.jpg", self.photo_bytes(message["photo"]), "image/jpeg")}
        elif not text:
            self.send_text(chat_id, "Send me text or a photo.")
            return
        self.call("sendChatAction", chat_id=chat_id, action="typing")
        data = {"message": text, "conversation_id": self.conversation(chat_id)}
        try:
            reply = self.api.post("/api/v1/chat", data=data, files=files, headers=self.user(message)).raise_for_status().json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:  # a limit: the API explains it in words
                self.send_text(chat_id, exc.response.json().get("detail") or "You've reached a limit. Try again later.")
                return
            logger.warning("chat request failed: %s", exc)
            self.send_text(chat_id, "Sorry, something went wrong. Try again in a moment.")
            return
        except httpx.HTTPError as exc:
            logger.warning("chat request failed: %s", exc)
            self.send_text(chat_id, "Sorry, something went wrong. Try again in a moment.")
            return
        self.deliver(chat_id, reply)
        if (reply.get("fetching") or reply.get("uploading")) and reply["conversation_id"] not in self.watching:
            self.watching.add(reply["conversation_id"])
            threading.Thread(
                target=self.wait_for_download, args=(chat_id, reply["conversation_id"], self.user(message)), daemon=True
            ).start()

    def usage_text(self, message: dict[str, Any]) -> str:
        try:
            u = self.api.get("/api/v1/me/usage", headers=self.user(message)).raise_for_status().json()
        except httpx.HTTPError:
            return "Sorry, I couldn't check your usage right now."
        hours, minutes = u["resets_in_sec"] // 3600, u["resets_in_sec"] % 3600 // 60
        return (
            f"Today you've used {u['tokens_used']:,} of your {u['tokens_limit']:,} tokens ({u['tokens_left']:,} left), "
            f"{u['uploads_used']} of {u['uploads_limit']} video uploads, and {u['pexels_used']} of {u['pexels_limit']} "
            f"stock-footage downloads. Everything resets in {hours} h {minutes} min."
        )

    def deliver(self, chat_id: int, reply: dict[str, Any]) -> None:
        clips = reply.get("clips") or []
        if not clips:
            self.send_text(chat_id, reply["reply"])
            return
        if clips[0].get("explanation") != reply["reply"]:  # the answer text differs from the clip's caption
            self.send_text(chat_id, reply["reply"])
        for clip in clips:
            self.call("sendChatAction", chat_id=chat_id, action="upload_video")
            self.send_clip(chat_id, clip)

    def wait_for_download(
        self,
        chat_id: int,
        conversation_id: str,
        user: dict[str, str] | None = None,
        every_sec: float = 10,
        give_up_sec: float = 45 * 60,
    ) -> None:
        """Poll until the upload (or Pexels download) is processed, keeping ONE status message up to date, so a long
        wait never looks like a dead bot. Then deliver the result (and any request that was waiting for it)."""
        started = time.time()
        status_id = self.send_text(chat_id, "⏳ Processing…")
        shown = None
        try:
            while time.time() - started < give_up_sec:
                time.sleep(every_sec)
                try:
                    update = self.api.get(f"/api/v1/chat/{conversation_id}/updates", headers=user).raise_for_status().json()
                except httpx.HTTPError:
                    continue
                elapsed = time.time() - started
                if update["status"] == "pending":
                    text = f"⏳ {update.get('status_text') or self._download_progress(update.get('progress') or {}, elapsed)}"
                    if text != shown:
                        self.edit_text(chat_id, status_id, text)
                        shown = text
                    continue
                done = "✅" if update["status"] == "done" and "couldn't process" not in (update.get("reply") or "") else "⚠️"
                self.edit_text(chat_id, status_id, f"{done} Processed in {int(elapsed // 60)}:{int(elapsed % 60):02d}")
                if update.get("reply"):
                    self.deliver(chat_id, update)
                return
            self.edit_text(chat_id, status_id, "⚠️ Still processing after a long time. Ask me again later.")
        finally:
            self.watching.discard(conversation_id)

    @staticmethod
    def _download_progress(progress: dict[str, int], elapsed: float) -> str:
        total = sum(progress.values()) or 1
        clock = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        return f"Downloading and processing videos from Pexels: {progress.get('ready', 0)} of {total} ready · {clock}"

    # ---- the loop ----------------------------------------------------------------------------------------------------
    def run(self) -> None:
        me = self.call("getMe")
        logger.info("running as @%s", me["username"])
        offset = None
        while True:
            try:
                updates = self.call("getUpdates", timeout=50, offset=offset, allowed_updates=["message"])
            except (httpx.HTTPError, RuntimeError) as exc:
                logger.warning("getUpdates failed, retrying: %s", exc)
                time.sleep(5)
                continue
            for update in updates:
                offset = update["update_id"] + 1
                if "message" in update:
                    # One thread per message: a slow turn (or a download wait) doesn't block other chats.
                    threading.Thread(target=self._safe_handle, args=(update["message"],), daemon=True).start()

    def _safe_handle(self, message: dict[str, Any]) -> None:
        try:
            self.handle(message)
        except Exception:
            logger.exception("message handling failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines include the URL, i.e. the bot token
    settings = Settings()
    check_production(settings)
    if settings.telegram_bot_token:
        for handler in logging.getLogger().handlers:
            handler.addFilter(RedactToken(settings.telegram_bot_token))
    if not settings.telegram_bot_token:
        logger.warning("TELEGRAM_BOT_TOKEN is not set: the bot is off")  # exit 0: compose doesn't restart it
        return
    allowed = {int(x) for x in settings.telegram_allowed_chat_ids.split(",") if x.strip()}
    storage = make_storage_client(settings)
    TelegramBot(settings.telegram_bot_token, settings.bot_api_url, storage, allowed, settings.service_token).run()


if __name__ == "__main__":
    main()
