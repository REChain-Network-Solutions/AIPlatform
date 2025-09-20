@if ($app_is_not_demo && auth()->check())
    <div id="vip-intercom-container">
    </div>

    <script>
        (() => {
            fetch('/vip-intercom-partial')
                .then(response => response.text())
                .then(html => {
                    const container = document.getElementById('vip-intercom-container');
                    if (container) {
                        container.innerHTML = html;
                    }

                    const temp = document.createElement('div');
                    temp.innerHTML = html;
                    const scripts = temp.getElementsByTagName('script');

                    for (let script of scripts) {
                        const newScript = document.createElement('script');
                        if (script.src) {
                            newScript.src = script.src;
                        } else {
                            newScript.textContent = script.textContent;
                        }
                        document.body.appendChild(newScript);
                    }
                });

            document.addEventListener('alpine:init', () => {
                Alpine.data('externalChatbotSupport', () => ({
                    windowState: 'close',
                    currentView: 'conversation-messages',

                    init() {
                        this.windowState = this.$el.getAttribute('data-window-state');
                        this.toggleWindowState = this.toggleWindowState.bind(this);
                        this.onSendMessage = this.onSendMessage.bind(this);
                        this.scrollMessagesToBottom = this.scrollMessagesToBottom.bind(this);
                    },

                    toggleWindowState(state) {
                        if (state === this.windowState) return;
                        this.windowState = state ? state : (this.windowState === 'open' ? 'close' : 'open');
                        this.$el.setAttribute('data-window-state', this.windowState);
                    },

                    toggleView(view) {
                        if (view === this.currentView) return;
                        this.currentView = view;
                    },

                    onMessageFieldHitEnter(event) {
                        if (!event.shiftKey) {
                            this.onSendMessage();
                        } else {
                            event.target.value += '\n';
                            event.target.scrollTop = event.target.scrollHeight;
                        }
                    },

                    onMessageFieldInput(event) {
                        const messageString = this.$refs.message.value.trim();
                        if (messageString) {
                            this.$refs.submitBtn.removeAttribute('disabled');
                        } else {
                            this.$refs.submitBtn.setAttribute('disabled', 'disabled');
                        }
                    },

                    openConversation(conversationId, fetchData = false) {
                        return this.toggleView('conversation-messages');
                    },

                    onSendMessage() {
                        const messageString = this.$refs.message.value.trim();
                        if (!messageString) return;

                        this.$refs.message.value = '';
                        this.$refs.submitBtn.setAttribute('disabled', 'disabled');

                        this.addDemoMessage(messageString, 'user', new Date().toLocaleTimeString());

                        setTimeout(() => {
                            this.addDemoMessage(
                                '',
                                'assistant',
                                new Date().toLocaleTimeString());
                        }, 800);
                    },

                    addDemoMessage(content, role, time) {
                        const templateSelector = role === 'user' ? '#lqd-ext-chatbot-premium-user-msg-temp' : '#lqd-ext-chatbot-premium-assistant-msg-temp';
                        const messageTemplate = document.querySelector(templateSelector).content.cloneNode(true);
                        const contentEl = messageTemplate.querySelector(
                            `.lqd-ext-chatbot-premium-window-conversation-message-content ${role === 'user' ? 'p' : 'div'}`);
                        const timeEl = messageTemplate.querySelector('.lqd-ext-chatbot-premium-window-conversation-message-time');

                        if (role === 'user') {
                            contentEl.innerText = content;
                        } else {
                            // contentEl.innerHTML = content;
                        }

                        timeEl.innerText = time;

                        this.$refs.conversationMessages.appendChild(messageTemplate);
                        this.scrollMessagesToBottom(true);
                        this.animateMessage(this.$refs.conversationMessages.lastElementChild);
                    },

                    animateMessage(messageElement) {
                        messageElement.animate([{
                                transform: 'translateY(3px)',
                                opacity: 0
                            },
                            {
                                transform: 'translateY(0)',
                                opacity: 1
                            }
                        ], {
                            duration: 150,
                            easing: 'ease'
                        });
                    },

                    scrollMessagesToBottom(smooth = false) {
                        if (this.$refs.conversationMessages) {
                            this.$refs.conversationMessages.scrollTo({
                                top: this.$refs.conversationMessages.scrollHeight,
                                behavior: smooth ? 'smooth' : 'auto'
                            });
                        }
                    }
                }));
            });
        })();
    </script>
@endif
