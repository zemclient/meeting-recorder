module.exports = {
  version: "1.0.0",
  title: "Meeting Recorder — запись встреч, задачи и сроки",
  description: "Записывает аудио встреч, расшифровывает речь (локально через faster-whisper или через OpenAI/Yandex) и извлекает задачи, сроки и итог локальной LLM (llama.cpp) либо OpenAI. Работает оффлайн без сторонних серверов.",
  icon: "https://avatars.githubusercontent.com/u/0?s=200",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env") || info.exists("app/requirements.txt")
    let running = info.running("start.js")

    if (running) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Открыть интерфейс",
          href: local.url,
        }, {
          icon: "fa-solid fa-terminal",
          text: "Терминал",
          href: "start.js",
        }]
      }
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Терминал",
        href: "start.js",
      }]
    }

    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Установить",
        href: "install.js",
      }]
    }

    return [{
      default: true,
      icon: "fa-solid fa-play",
      text: "Запустить",
      href: "start.js",
    }, {
      icon: "fa-solid fa-arrows-rotate",
      text: "Обновить зависимости",
      href: "update.js",
    }, {
      icon: "fa-solid fa-trash",
      text: "Сбросить",
      href: "reset.js",
    }, {
      icon: "fa-solid fa-download",
      text: "Переустановить",
      href: "install.js",
    }]
  }
}
