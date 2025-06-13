// Global variables
let isWaiting = false;
let sessionId = 'session_' + Math.random().toString(36).substring(2, 15);

// Add a message to the chat UI
function addMessage(message, isUser) {
    const messagesDiv = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    messageDiv.textContent = message;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const messagesDiv = document.getElementById('chat-messages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'message ai-message';
    typingDiv.textContent = 'Thinking...';
    messagesDiv.appendChild(typingDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingDiv = document.getElementById('typing-indicator');
    if (typingDiv) {
        typingDiv.remove();
    }
}

// Send message to the server
async function sendMessage() {
    if (isWaiting) return;

    const input = document.getElementById('user-input');
    const message = input.value.trim();
    
    // Validate input
    if (!message || message.length === 0) {
        addMessage('Error: Please enter a non-empty message.', false);
        return;
    }
    
    // Display user message
    addMessage(message, true);
    input.value = '';
    isWaiting = true;

    // Create payload matching ChatRequest schema
    const payload = {
        message: message,
        session_id: sessionId
        // timestamp omitted to avoid validation error (matches successful curl)
        // context omitted as it's optional and not needed
    };

    showTypingIndicator();
    
    try {
        // Log payload for debugging
        console.log('Sending payload:', payload);
        
        // Use full URL to ensure correct endpoint
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('Error response:', errorData); // Log server response
            throw new Error(`HTTP ${response.status}: ${errorData.detail || 'Unknown error'}`);
        }

        const data = await response.json();
        removeTypingIndicator();
        addMessage(data.response, false);
    } catch (error) {
        removeTypingIndicator();
        console.error('Chat error:', error);
        addMessage(`Error: ${error.message || 'Failed to communicate with the server. Please try again.'}`, false);
    } finally {
        isWaiting = false;
    }
}

// Event Listeners
document.getElementById('user-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});


document.getElementById('send-button')?.addEventListener('click', sendMessage);

// Debug: Test payload with /chat/debug endpoint (uncomment to use)
// async function debugRequest() {
//     const payload = {
//         message: "Hello",
//         session_id: sessionId
//     };
//     try {
//         const response = await fetch('http://localhost:8000/chat/debug', {
//             method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             },
//             body: JSON.stringify(payload)
//         });
//         const debugData = await response.json();
//         console.log('Debug response:', debugData);
//     } catch (error) {
//         console.error('Debug error:', error);
//     }
// }
// debugRequest();