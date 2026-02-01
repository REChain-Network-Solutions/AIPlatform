<div
    class="lqd-ext-chatbot-history invisible fixed end-0 start-0 top-0 z-10 flex h-screen bg-background opacity-0 transition-all max-sm:block lg:start-[--navbar-width] [&.lqd-open]:visible [&.lqd-open]:opacity-100"
    x-data="externalChatbotHistory"
    :class="{ 'lqd-open': open }"
    @keydown.window.escape="setOpen(false)"
>
    <div
        class="lqd-ext-chatbot-history-sidebar group/history-sidebar relative w-full shrink-0 space-y-5 border-e px-6 py-7 sm:h-full sm:w-[440px] sm:overflow-y-auto"
        :class="{ 'mobile-dropdown-open': mobileDropdownOpen }"
    >
        <x-button
            class="text-2xs font-medium opacity-65 hover:opacity-100 max-sm:hidden"
            variant=link
            @click.prevent="setOpen(false)"
        >
            <x-tabler-chevron-left class="size-4" />
            @lang('Back to dashboard')
        </x-button>
        <x-button
            class="!mt-0 ms-auto flex size-9 border text-2xs font-medium text-heading-foreground opacity-65 hover:opacity-100 sm:hidden"
            variant=link
            @click.prevent="setOpen(false)"
            title="{{ __('Back to dashboard') }}"
        >
            <x-tabler-x class="size-4" />
        </x-button>

        <x-button
            class="w-full rounded-none border-y py-4 sm:hidden"
            variant="link"
            @click="mobileDropdownOpen = !mobileDropdownOpen"
        >
            @lang('Conversations List')
            <x-tabler-chevron-down class="ms-auto size-4" />
        </x-button>

        <div
            class="space-y-5 max-sm:absolute max-sm:inset-x-0 max-sm:top-full max-sm:z-2 max-sm:!m-0 max-sm:hidden max-sm:h-[60vh] max-sm:overflow-y-auto max-sm:bg-background max-sm:px-5 max-sm:pb-5 max-sm:shadow-xl max-sm:group-[&.mobile-dropdown-open]/history-sidebar:block">
            <h3>
                @lang('All Chats')
            </h3>

            <form
                class="hidden"
                action="#"
                @submit.prevent="handleSearch"
            >
                <x-forms.input
                    class="w-full rounded-full ps-10 text-2xs font-medium"
                    type="search"
                    name="search"
                    placeholder="{{ __('Search for anything...') }}"
                    x-ref="historySearchInput"
                >
                    <x-tabler-search class="absolute start-4 top-1/2 size-4 -translate-y-1/2" />
                </x-forms.input>
            </form>

            @include('chatbot::home.chats-history.chats-list')

            <div
                class="lqd-ext-chatbot-history-load-wrap grid place-items-center font-medium text-heading-foreground"
                x-ref="loadMoreWrap"
            >
                <x-button
                    class="lqd-ext-chatbot-history-load-more col-start-1 col-end-1 row-start-1 row-end-1 w-full"
                    variant="link"
                    href="{{ route('dashboard.chatbot.conversations.with.paginate', ['page' => 1]) }}"
                    x-ref="loadMore"
                    x-show="!allLoaded && !fetching"
                >
                    {{ __('Load More...') }}
                </x-button>
                <span
                    class="lqd-ext-chatbot-history-loading col-start-1 col-end-1 row-start-1 row-end-1 inline-flex gap-2"
                    x-show="!allLoaded && fetching"
                    x-ref="loading"
                >
                    {{ __('Loading') }}
                    <x-tabler-refresh class="size-4 animate-spin" />
                </span>
                <span
                    class="lqd-ext-chatbot-history-all-loaded col-start-1 col-end-1 row-start-1 row-end-1 inline-flex gap-2"
                    x-ref="allLoaded"
                    x-show="allLoaded"
                >
                    {{ __('All Items Loaded') }}
                    <x-tabler-check class="size-4" />
                </span>
            </div>
        </div>
    </div>

    <div class="lqd-ext-chatbot-history-content-wrap flex grow flex-col max-sm:h-[70vh]">
        <div class="lqd-ext-chatbot-history-head hidden h-[--header-height] w-full justify-between gap-4 border-b px-8 py-5">
            <form
                class="flex grow items-center gap-3"
                action="#"
                @submit.prevent="handleChangeTitle"
            >
                <x-forms.input
                    class="border-none bg-transparent bg-none p-0 font-heading text-base font-semibold"
                    containerClass="grow"
                    type="text"
                    name="title"
                    placeholder="{{ __('Search for anything...') }}"
                    x-ref="historyChatTitleInput"
                    value="Chat873: Can’t delete account"
                />
                <x-button
                    class="inline-grid size-9 place-content-center shadow-[0_0_36px_rgba(0,0,0,0.15)]"
                    size="none"
                    variant="none"
                    type="submit"
                    title="{{ __('Change Name') }}"
                >
                    <svg
                        width="18"
                        height="18"
                        viewBox="0 0 18 18"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M6.81975 15.3629L1.35547 16.8929L2.88547 11.4286L12.8912 1.47143C13.0043 1.35577 13.1393 1.26388 13.2884 1.20114C13.4375 1.1384 13.5976 1.10608 13.7594 1.10608C13.9212 1.10608 14.0813 1.1384 14.2304 1.20114C14.3795 1.26388 14.5145 1.35577 14.6276 1.47143L16.7769 3.63286C16.8906 3.74582 16.9808 3.88015 17.0423 4.02811C17.1039 4.17607 17.1356 4.33475 17.1356 4.495C17.1356 4.65525 17.1039 4.81393 17.0423 4.96189C16.9808 5.10985 16.8906 5.24418 16.7769 5.35714L6.81975 15.3629Z"
                            stroke="currentColor"
                            stroke-width="1.7"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </x-button>
            </form>

            <div class="ms-auto flex grow items-center justify-end gap-2">
                <x-button
                    variant="ghost-shadow"
                    @click.prevent="handleSummarize"
                >
                    @lang('Summarize')
                </x-button>

                <x-dropdown.dropdown
                    class:dropdown-dropdown="max-lg:end-auto max-lg:start-0"
                    anchor="end"
                    offsetY="20px"
                >
                    <x-slot:trigger
                        class="size-9"
                        variant="none"
                        title="{{ __('More Options') }}"
                    >
                        <svg
                            width="3"
                            height="14"
                            viewBox="0 0 3 14"
                            fill="currentColor"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M3 12C3 12.8 2.3 13.5 1.5 13.5C0.7 13.5 0 12.8 0 12C0 11.2 0.7 10.5 1.5 10.5C2.3 10.5 3 11.2 3 12ZM3 7C3 7.8 2.3 8.5 1.5 8.5C0.7 8.5 0 7.8 0 7C0 6.2 0.7 5.5 1.5 5.5C2.3 5.5 3 6.2 3 7ZM3 2C3 2.8 2.3 3.5 1.5 3.5C0.7 3.5 0 2.8 0 2C0 1.2 0.7 0.5 1.5 0.5C2.3 0.5 3 1.2 3 2Z"
                            />
                        </svg>
                    </x-slot:trigger>
                    <x-slot:dropdown
                        class="min-w-48 text-xs font-medium"
                    >
                        <ul>
                            <li>
                                <a
                                    class="flex items-center gap-2 rounded-md px-3 py-2 transition-colors hover:bg-red-500 hover:text-white"
                                    href="#"
                                    @click.prevent="handleDelete"
                                >
                                    <x-tabler-trash class="size-4" />
                                    @lang('Delete')
                                </a>
                            </li>
                        </ul>
                    </x-slot:dropdown>
                </x-dropdown.dropdown>
            </div>
        </div>
        <div class="lqd-ext-chatbot-history-messages-wrap grow overflow-hidden">
            <div
                class="lqd-ext-chatbot-history-messages flex h-full flex-col gap-2 overflow-y-auto p-10"
                x-ref="historyMessages"
            >
                @include('chatbot::home.chats-history.chat-messages')
            </div>
        </div>

        <div
            class="invisible absolute inset-0 z-1 grid place-items-center opacity-0 backdrop-blur-md transition-all"
            :class="{ 'opacity-0': !firstLoading, 'invisible': !firstLoading }"
        >
            <x-tabler-loader-2 class="size-8 animate-spin text-primary" />
        </div>
    </div>
</div>

@push('script')
    <script>
        (() => {
            document.addEventListener('alpine:init', () => {
                Alpine.data('externalChatbotHistory', () => ({
                    open: false,
                    chatsList: [],
                    activeChat: null,
                    fetching: true,
                    lastTimeFetch: null,
                    currentPage: 1,
                    allLoaded: false,
                    /**
                     * @type {IntersectionObserver}
                     */
                    loadMoreIO: null,
                    mobileDropdownOpen: false,
                    firstLoading: true,
                    init() {
                        Alpine.store('externalChatbotHistory', this);
                    },
                    setupLoadMoreIO() {
                        const load = async (entry) => {
                            this.currentPage += 1;
                            this.$refs.loadMore.href = this.$refs.loadMore.href.replace(/page=\d+/, `page=${this.currentPage}`);

                            await this.fetchChats();
                        };

                        this.loadMoreIO = new IntersectionObserver(async ([entry], observer) => {
                            if (entry.isIntersecting && !this.fetching && !this.allLoaded) {
                                await load();

                                if (entry.isIntersecting && !this.fetching && !this.allLoaded) {
                                    await load();
                                }
                            }
                        });

                        this.loadMoreIO.observe(this.$refs.loadMoreWrap);
                    },
                    async setOpen(open) {
                        if (this.open === open) return;

                        const topNoticeBar = document.querySelector('.top-notice-bar');
                        const navbar = document.querySelector('.lqd-navbar');
                        const pageContentWrap = document.querySelector('.lqd-page-content-wrap');
                        const navbarExpander = document.querySelector('.lqd-navbar-expander');

                        this.open = open;

                        document.documentElement.style.overflow = this.open ? 'hidden' : '';

                        if (navbar) {
                            navbar.style.position = this.open ? 'fixed' : '';
                        }

                        if (pageContentWrap) {
                            pageContentWrap.style.paddingInlineStart = this.open ? 'var(--navbar-width)' : '';
                        }

                        if (topNoticeBar) {
                            topNoticeBar.style.visibility = this.open ? 'hidden' : '';
                        }

                        if (navbarExpander) {
                            navbarExpander.style.visibility = this.open ? 'hidden' : '';
                            navbarExpander.style.opacity = this.open ? 0 : 1;
                        }

                        if (this.open) {
                            await this.fetchChats();
                            this.setupLoadMoreIO();
                        }
                    },
                    async fetchChats() {
                        if (this.allLoaded) {
                            return this.fetching = false;
                        }

                        this.fetching = true;

                        const res = await fetch(this.$refs.loadMore.href, {
                            method: 'GET',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                            },
                        });
                        const data = await res.json();
                        const {
                            data: conversations
                        } = data;

                        if (!res.ok || !conversations) {
                            if (data.message) {
                                toastr.error(data.message);
                            }
                            return;
                        }

                        this.lastTimeFetch = new Date().getTime();

                        this.chatsList.push(...conversations.reverse());

                        if (!this.activeChat && conversations.length) {
                            this.activeChat = conversations[0].id;
                        }

                        if (this.currentPage >= data.meta.last_page) {
                            this.allLoaded = true;
                        }

                        this.fetching = false;

                        this.firstLoading = false;

                        this.scrollMessagesToBottom();
                    },
                    async handleSearch() {
                        // TODO: Implement search logic
                    },
                    async handleChangeTitle() {
                        // TODO: Implement change title logic
                    },
                    async handleDelete() {
                        // TODO: Implement delete logic
                    },
                    async handleSummarize() {
                        // TODO: Implement delete logic
                    },
                    async setActiveChat(event) {
                        const triggerEl = event.currentTarget;
                        const chatId = triggerEl.getAttribute('data-id');
                        const triggerParent = triggerEl.closest('li');

                        if (!chatId) return;

                        this.activeChat = chatId;

                        const parentSiblings = Array.from(triggerParent.parentNode.children).filter(child => child !== triggerParent);
                        parentSiblings.forEach(sibling => sibling.classList.remove('lqd-active'));

                        triggerParent.classList.add('lqd-active');

                        this.scrollMessagesToBottom();
                    },
                    scrollMessagesToBottom() {
                        this.$nextTick(() => {
                            this.$refs.historyMessages.scrollTo({
                                top: this.$refs.historyMessages.scrollHeight,
                                behavior: 'smooth',
                            });
                        })
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
                                // const codeString = str.replace(/&/g, '&amp;').replace(/</g, '&lt;');
                                const codeString = str;

                                if (Prism.languages[language]) {
                                    const highlighted = Prism.highlight(codeString, Prism.languages[language], language);
                                    return `<pre class="language-${language}"><code data-lang="${language}" class="language-${language}">${highlighted}</code></pre>`;
                                }

                                return codeString;
                            }
                        });

                        renderer.use(function(md) {
                            md.core.ruler.after('inline', 'convert_links', function(state) {
                                state.tokens.forEach(function(blockToken) {
                                    if (blockToken.type !== 'inline') return;
                                    blockToken.children.forEach(function(token, idx) {
                                        if (token.content.includes('<a ')) {
                                            const linkRegex = /(.*)(<a\s+href="([^"]+)"[^>]*>([^<]+)<\/a>)(.*)/;
                                            const linkMatch = token.content.match(linkRegex);

                                            if (linkMatch) {
                                                const [, before, , href, text, after] = linkMatch;

                                                const beforeToken = new state.Token('text', '', 0);
                                                beforeToken.content = before;

                                                const newToken = new state.Token('link_open', 'a', 1);
                                                newToken.attrs = [
                                                    ['href', href]
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
@endpush
