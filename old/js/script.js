document.addEventListener("DOMContentLoaded", function () {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    // Function to add message to chat
    function addMessage(text, sender) {
        let messageDiv = document.createElement("div");
        messageDiv.classList.add("message");
        messageDiv.classList.add(sender === "user" ? "user-message" : "ai-message");
        messageDiv.innerText = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to handle user input
    function handleInput() {
        let userText = userInput.value.trim();
        if (userText === "") return;

        addMessage(userText, "user");
        userInput.value = "";

        // Simulated AI response
        setTimeout(() => {
            let aiResponse = "Predicting for " + userText + "... (This is a placeholder response)";
            addMessage(aiResponse, "ai");
        }, 1000);
    }

    // Event Listeners
    sendBtn.addEventListener("click", handleInput);
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") handleInput();
    });
});
