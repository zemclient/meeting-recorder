module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        path: "app",
        message: [
          "rm -rf env",
          "uv venv env",
          "uv pip install -r requirements.txt"
        ]
      }
    }
  ]
}
