<script>
    (() => {
        document.addEventListener('alpine:init', () => {
            Alpine.data('externalChatbot', (isEditor = false) => ({
                conversations: [],
                activeConversationData: null,
                showConnectButton: true,
				showConnectButtonStepTwo: false,
				showConnectButtonStepOne: true,
                connect_agent_at: null,
                messages: [],
                activeConversation: null,
                fetching: false,
                windowState: 'close',
                assistantMessageBubbles: {},
                currentView: '{{ $is_editor ? 'conversation-messages' : 'conversations-list' }}',
                ablyInstance: null,
				ablyChannel: null,
                init() {
                    this.windowState = this.$el.getAttribute('data-window-state');

                    this.handleWindowMessages = this.handleWindowMessages.bind(this);
                    this.toggleWindowState = this.toggleWindowState.bind(this);
                    this.onSendMessage = this.onSendMessage.bind(this);
                    this.scrollMessagesToBottom = this.scrollMessagesToBottom.bind(this);
                    this.dontConnectToAgent = this.dontConnectToAgent.bind(this);

                    @if ($is_editor)
                        this.$data.externalChatbot = this;
                    @else
                        this.getSession();
                        if (window.self !== window.top) {
                            window.addEventListener("message", this.handleWindowMessages);
                            document.documentElement.classList.add('lqd-ext-chatbot-embedded');
                        }
                    @endif
                },
				async initAbly() {

					@if(isset($session) && setting('ably_public_key'))
						if (!this.ablyInstance){
							this.ablyInstance = new Ably.Realtime.Promise("{{ setting('ably_public_key') }}");
						}

						let channelName = 'conversation-session-{{ $session }}';

						this.ablyChannel = this.ablyInstance.channels.get(channelName);

						await this.ablyChannel.subscribe('new-message', (message) => {
							let data = message.data;

							let incomingConversationId = data.conversationId;

							if (incomingConversationId === this.activeConversation) {
								this.messages.push(data.history);

								this.$nextTick(this.scrollMessagesToBottom);
							}

							this.conversations.map((conversation) => {
								if (conversation.id === incomingConversationId) {
									conversation.last_message = data.history.message;
								}
								return conversation;
							});
						});
					@endif
                },
				connectToAgentStepOne() {
					this.showConnectButtonStepOne = false;
					this.showConnectButtonStepTwo = true;
				},
				conversationMessages() {

					this.showConnectButtonStepOne = true;
					this.showConnectButtonStepTwo = false;

					return this.messages;
				},
                connectToAgent() {

                    @if (isset($chatbot) && isset($session) && \App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent'))

                        let route = '{{ route('api.v2.chatbot.conversion.connect.support', [$chatbot->uuid, $session]) }}';

                        fetch(route, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-timezone': Intl.DateTimeFormat().resolvedOptions().timeZone
                            },
                            body: JSON.stringify({
                                conversation_id: this.activeConversation,
                            })
                        }).then((res) => {
                            return res.json();
                        }).then((data) => {
                            if (data.history) {
                                this.messages.push(data.history);
                            }

                            this.$nextTick(this.scrollMessagesToBottom);

                            localStorage.setItem('connectToAgentStore:' + this.activeConversationData?.id, 'on');

                            this.showConnectButton = false;

                            this.connect_agent_at = data.connect_agent_at;

                            this.activeConversationData = data.data;

                            this.conversations = this.conversations.map((conversation) => {

                                if (conversation.id === data.data.id) {
                                    conversation = data.data;
                                }

                                return conversation;
                            });

							this.showConnectButtonStepOne = true;
							this.showConnectButtonStepTwo = false;
                        }).catch((err) => {
                        });
                    @endif
                },
                dontConnectToAgent() {
                    localStorage.setItem('connectToAgentStore:' + this.activeConversationData?.id, 'off');

                    this.showConnectButton = false;
                },
                handleWindowMessages(event) {
                    switch (event.data.type) {
                        case 'lqd-ext-chatbot-request-styling':
                            this.handleStylingResponse(event);
                            break;
                    }
                },

                handleStylingResponse(event) {
                    const chatbotElStyles = getComputedStyle(this.$el);
                    const styles = {};
                    const attrs = {};
                    [
                        '--lqd-ext-chat-font-family',
                        '--lqd-ext-chat-offset-y',
                        '--lqd-ext-chat-offset-x',
                        '--lqd-ext-chat-trigger-w',
                        '--lqd-ext-chat-trigger-h',
                        '--lqd-ext-chat-window-w',
                        '--lqd-ext-chat-window-h',
                        '--lqd-ext-chat-window-y-offset',
                        '--lqd-ext-chat-primary',
                        '--lqd-ext-chat-primary-foreground',
                    ].forEach(attr => styles[attr] = chatbotElStyles.getPropertyValue(attr) || '');

                    ['data-pos-x', 'data-pos-y'].forEach(attr => attrs[attr] = this.$el.getAttribute(attr));

                    event.source.postMessage({
                            type: 'lqd-ext-chatbot-response-styling',
                            data: {
                                styles,
                                attrs
                            },
                        },
                        event.origin,
                    );
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
                        event.target.scrollTop = event.target.scrollHeight
                    };
                },
                onMessageFieldInput(event) {
                    const messageString = this.$refs.message.value.trim();

                    if (messageString) {
                        this.$refs.submitBtn.removeAttribute('disabled');
                    } else {
                        this.$refs.submitBtn.setAttribute('disabled', 'disabled');
                    }
                },
                async getSession() {
                    this.fetching = true;

                    const res = await fetch('{{ isset($routes) ? $routes['getSession'] : '' }}');
                    const data = await res.json();

                    if (!res.ok) {
                        console.error(data);
                        return this.fetching = false;
                    }

                    this.conversations = data.conversations;

                    this.activeChatbot = data.data;

                    if (!this.conversations.length) {
                        await this.startNewConversation();
                    } else {
						@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent') && isset($chatbot) && isset($session))
							await this.initAbly();
						@endif
					}

                    this.fetching = false;
                },
                async startNewConversation(fetchData = {{ $is_editor ? 'false' : 'true' }}) {
                    if (!fetchData) {
                        return this.toggleView('conversation-messages');
                    }

                    this.fetching = true;

                    this.assistantMessageBubbles = {};

                    const res = await fetch(`{{ isset($routes) ? $routes['conversations'] : '' }}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            prompt: this.activeChatbot.bubble_message
                        })
                    });


                    const data = await res.json();

                    if (!res.ok) {

						if(this.ablyChannel && this.ablyInstance) {
							this.ablyChannel.unsubscribe();
							this.ablyChannel = null;
						}

						@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent') && isset($chatbot) && isset($session))
							await this.initAbly();
						@endif

                        return this.fetching = false;
                    }

                    this.conversations.push(data.data);

                    await this.$nextTick();

                    this.openConversation(data.data.id);

                    this.showConnectButton = false;

                    this.toggleView('conversation-messages');

                    this.fetching = false;


					@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('chatbot-agent') && isset($chatbot) && isset($session))
						if(this.ablyChannel && this.ablyInstance) {
							this.ablyChannel.unsubscribe();
							// this.ablyInstance.close();
							this.ablyChannel = null;
						}
						await this.initAbly();
					@endif

				},
                async openConversation(conversationId, fetchData = {{ $is_editor ? 'false' : 'true' }}) {

                    if (!fetchData) {
                        return this.toggleView('conversation-messages');
                    }

                    this.fetching = true;

                    this.assistantMessageBubbles = {};

                    this.activeConversationData = this.conversations.find(conversation => conversation.id === conversationId);

                    // this.connect_agent_at = this.activeConversationData?.connect_agent_at;

					this.connect_agent_at = null;

                    this.activeConversation = this.activeConversationData?.id;

                    // let connectToAgentStore = localStorage.getItem('connectToAgentStore:' + this.activeConversationData?.id) ?? null;
					//
                    // if (this.activeConversationData?.connect_agent_at === null && connectToAgentStore === 'on') {
                    //     this.showConnectButton = false;
                    // } else if (this.activeConversationData?.connect_agent_at && connectToAgentStore === null) {
                    //     localStorage.setItem('connectToAgentStore:' + this.activeConversationData?.id, 'on');
                    // } else if (this.activeConversationData?.connect_agent_at && connectToAgentStore === 'off') {
                    //     this.showConnectButton = false;
                    // }

                    if (!this.activeConversation) {
                        console.error('Conversation not found');
                        return this.fetching = false;
                    }

                    const res = await fetch(`{{ isset($routes) ? $routes['conversations'] : '' }}/${conversationId}/messages`, {
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json',
                            'X-timezone': Intl.DateTimeFormat().resolvedOptions().timeZone
                        },
                    });
                    const data = await res.json();

                    if (!res.ok) {
                        console.error(data);
                        return this.fetching = false;
                    }

                    this.messages = data.data.reverse();

                    await this.$nextTick();

                    this.toggleView('conversation-messages');

                    this.$nextTick(this.scrollMessagesToBottom);

                    this.fetching = false;
                },
                async onSendMessage() {
                    const messageString = this.$refs.message.value.trim();

                    if (!messageString) return;

                    this.$refs.message.value = '';
                    this.$refs.submitBtn.setAttribute('disabled', 'disabled');

                    @if ($is_editor)
                        this.addDemoMessage(messageString, 'user', new Date().toLocaleTimeString());
                        // echo the message in editor
                        const timeout = setTimeout(() => {
                            this.addDemoMessage(messageString, 'assistant', new Date().toLocaleTimeString());
                            clearTimeout(timeout);
                        }, 200);
                    @else
                        if (!this.activeConversation) {
                            console.error('No active conversation');
                            return;
                        }

                        const conversation = this.conversations.find(conversation => conversation.id === this.activeConversation);

                        const newUserMessage = {
                            id: new Date().getTime(),
                            message: messageString,
                            role: 'user',
                            created_at: new Date().toLocaleString()
                        };

                        this.messages.push(newUserMessage);

                        let loaderMessage = null;

                        if (conversation.connect_agent_at == null) {
                            loaderMessage = {
                                role: 'loader',
                                id: `response-for-${newUserMessage.id}`,
                                created_at: new Date().toLocaleString()
                            };

                            this.messages.push(loaderMessage);
                        }

                        this.$nextTick(this.scrollMessagesToBottom);

                        const res = await fetch(`{{ isset($routes) ? $routes['conversations'] : '' }}/${this.activeConversation}/messages`, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-timezone': Intl.DateTimeFormat().resolvedOptions().timeZone
                            },
                            body: JSON.stringify({
                                prompt: messageString,
                            })
                        });

                        if (!res.ok) {

                            if (loaderMessage == null) {
                                const loaderMessage = {
                                    role: 'loader',
                                    id: `response-for-${newUserMessage.id}`,
                                    created_at: new Date().toLocaleString()
                                };

                                this.messages.push(loaderMessage);
                            }

                            const errorData = await res.json();

                            this.messages = this.messages.filter(msg => msg.id !== loaderMessage.id);
                            this.messages.push({
                                id: new Date().getTime(),
                                message: errorData.message ||
                                    '{{ __('Sorry, I could not process your request at the moment. Please try again later.') }}',
                                role: 'assistant',
                                created_at: new Date().toLocaleString()
                            });
                            this.$nextTick(this.scrollMessagesToBottom);
                            return;
                        }

                        const data = await res.json();

                        conversation.last_message = newUserMessage.message;
                        if (conversation.updated_at) {
                            conversation.updated_at = new Date().toLocaleString();
                        } else if (conversation.created_at) {
                            conversation.created_at = new Date().toLocaleString();
                        }

                        if (loaderMessage != null) {
                            this.onReceiveMessage(data.data, loaderMessage);
                        }
                        @if (isset($chatbot))
                            let condition = '{{ $chatbot->interaction_type->value }}';

                            if (conversation.connect_agent_at == null && condition ===
                                '{{ \App\Extensions\Chatbot\System\Enums\InteractionType::SMART_SWITCH }}') {
                                this.showConnectButton = true;
                            }
                        @endif
                    @endif
                },
                async onReceiveMessage(message, loaderMessage) {

                    @if ($is_editor)
                        this.addDemoMessage(message, 'assistant', new Date().toLocaleTimeString());
                    @else
                        const messageToReplace = this.messages.find(msg => msg.id === loaderMessage.id);
                        messageToReplace.role = 'assistant';
                        messageToReplace.message = message.message;
                        messageToReplace.created_at = new Date().toLocaleString();
                        messageToReplace.isNew = true;

                        this.$nextTick(this.scrollMessagesToBottom);
                    @endif
                },
                @if ($is_editor)
                    addDemoMessage(content, role, time) {
                            const templateSelector = role === 'user' ? '#lqd-ext-chatbot-user-msg-temp' : '#lqd-ext-chatbot-assistant-msg-temp';
                            const messageTemplate = document.querySelector(templateSelector).content.cloneNode(true);
                            const contentEl = messageTemplate.querySelector(
                                `.lqd-ext-chatbot-window-conversation-message-content ${role === 'user' ? 'p' : 'pre'}`);
                            const timeEl = messageTemplate.querySelector('.lqd-ext-chatbot-window-conversation-message-time');
                            const assistantAvatar = messageTemplate.querySelector('.lqd-ext-chatbot-window-conversation-message-avatar img');

                            contentEl.innerText = content;
                            timeEl.innerText = time;

                            if (assistantAvatar) {
                                assistantAvatar.src = this.activeChatbot ? `${window.location.origin}/${this.activeChatbot.avatar}` : '';
                            }

                            this.$refs.conversationMessages.appendChild(messageTemplate);
                            this.scrollMessagesToBottom(true);

                            this.animateMessage(this.$refs.conversationMessages.lastElementChild);
                        },
                @endif
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
                    this.$refs.conversationMessages.scrollTo({
                        top: this.$refs.conversationMessages.scrollHeight,
                        behavior: smooth ? 'smooth' : 'auto'
                    });
                },
                addMessage(message, el) {
                    const formattedMessage = this.getFormattedString(message);
                    const index = el.getAttribute('data-index');
                    const useTypeEffect = this.messages.find((msg, i) => i == index && msg.role === 'assistant' && msg.isNew);

                    this.assistantMessageBubbles[index] = el;

                    if (useTypeEffect) {
                        this.typeMessage(this.messages[index], index);
                        return '';
                    }

                    return formattedMessage;
                },
                typeMessage(messageObj, bubbleMessageIndex) {
                    const messageEl = this.assistantMessageBubbles[bubbleMessageIndex];
                    const messageString = messageObj.message;

                    if (!messageEl || !messageString) return '';

                    const formattedMessage = this.getFormattedString(messageString);

                    let i = 0;
                    const speed = 25; // typing speed in milliseconds
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = formattedMessage;
                    const textContent = tempDiv.innerHTML || '';
                    const words = textContent.split(' ');

                    messageObj.isNew = false;

                    const typeWriter = () => {
                        if (i < words.length) {
                            messageEl.innerHTML = words.slice(0, i + 1).join(' ') + ' ';
                            i++;
                            setTimeout(typeWriter, speed);
                            this.scrollMessagesToBottom();
                        }
                    };

                    typeWriter();
                },
                getFormattedString(string) {
                    if (!('markdownit' in window) || !string) return '';

                    string
                        .replace(/>(\s*\r?\n\s*)</g, '><')
                        .replace(/\n(?!.*\n)/, '');

                    const renderer = window.markdownit({
                        breaks: true,
                        highlight: (str, lang) => {
                            const language = lang && lang !== '' ? lang : 'md';
                            const codeString = str;

                            if (Prism.languages[language]) {
                                const highlighted = Prism.highlight(codeString, Prism.languages[language], language);
                                return `<pre class="language-${language}"><code data-lang="${language}" class="language-${language}">${highlighted}</code></pre>`;
                            }

                            return codeString;
                        }
                    });

                    renderer.use(function(md) {
                        md.core.ruler.after('inline', 'convert_elements', function(state) {
                            state.tokens.forEach(function(blockToken) {
                                if (blockToken.type !== 'inline') return;

                                let fullContent = '';

                                blockToken.children.forEach(token => {
                                    let {
                                        content,
                                        type
                                    } = token;


                                    switch (type) {
                                        case 'link_open':
                                            token.attrs?.push(['target', '_blank']);

                                            content =
                                                `<a ${token.attrs.map(([ key, value ]) => `${key}="${value}"`).join(' ')}>`;
                                            break;
                                        case 'link_close':
                                            content = '</a>';
                                            break;
                                    }

                                    fullContent += content;
                                });

                                if (fullContent.includes('<ol>') || fullContent.includes('<ul>')) {
                                    const listToken = new state.Token('html_inline', '', 0);
                                    listToken.content = fullContent.trim();
                                    listToken.markup = 'html';
                                    listToken.type = 'html_inline';

                                    blockToken.children = [listToken];
                                }
                            });
                        });

                        md.core.ruler.after('inline', 'convert_links', function(state) {
                            state.tokens.forEach(function(blockToken) {
                                if (blockToken.type !== 'inline') return;
                                blockToken.children.forEach(function(token, idx) {
                                    const {
                                        content
                                    } = token;
                                    if (content.includes('<a ')) {
                                        const linkRegex = /(.*)(<a\s+[^>]*\s+href="([^"]+)"[^>]*>([^<]*)<\/a>?)(.*)/;
                                        const linkMatch = content.match(linkRegex);

                                        if (linkMatch) {
                                            const [, before, , href, text, after] = linkMatch;

                                            const beforeToken = new state.Token('text', '', 0);
                                            beforeToken.content = before;

                                            const newToken = new state.Token('link_open', 'a', 1);
                                            newToken.attrs = [
                                                ['href', href],
                                                ['target', '_blank']
                                            ];
                                            const textToken = new state.Token('text', '', 0);
                                            textToken.content = text;
                                            const closingToken = new state.Token('link_close', 'a', -1);

                                            const afterToken = new state.Token('text', '', 0);
                                            afterToken.content = after;

                                            blockToken.children.splice(idx, 1, beforeToken, newToken, textToken,
                                                closingToken, afterToken);
                                        }
                                    }
                                });
                            });
                        });
                    });

                    return renderer.render(renderer.utils.unescapeAll(string));
                }
            }));
        });
    })();
</script>
