@push('script')
    <script src="{{ custom_theme_url('/assets/libs/markdownit/markdown-it.min.js') }}"></script>
    <script>
        document.addEventListener('alpine:init', () => {
            const UPDATES_DELIMITER = @json(\App\Services\Finance\AiPlanWizardService::UPDATES_DELIMITER);
            const ROUTES = {
                presets: @json(route('dashboard.admin.finance.plan.ai-wizard.presets')),
                validateStep: @json(route('dashboard.admin.finance.plan.ai-wizard.validate-step')),
                chat: @json(route('dashboard.admin.finance.plan.ai-wizard.chat')),
                store: @json(route('dashboard.admin.finance.plan.ai-wizard.store')),
            };
            const CSRF_TOKEN = @json(csrf_token());
            const DEFAULT_TOOLS = @json(array_fill_keys(array_column($planAiToolsMenu, 'key'), true));
            const DEFAULT_FEATURES = @json(array_fill_keys(array_column($planFeatureMenu, 'key'), true));
            const STEP_FIELDS = {
                basics: ['type', 'name', 'description', 'price', 'frequency', 'trial_days', 'active', 'is_featured'],
                credits: ['type', 'credit_system_type', 'shared_credits_amount', 'credit_tier', 'credit_limits'],
                features: ['type', 'features', 'plan_ai_tools', 'plan_features'],
            };
            const ALLOWED_UPDATE_FIELDS = @json(\App\Services\Finance\AiPlanWizardService::ALLOWED_FIELDS);
            const CREDIT_CATEGORIES = @json(\App\Services\Finance\AiPlanWizardService::CREDIT_CATEGORIES);

            const freshForm = () => ({
                type: 'subscription',
                name: '',
                description: '',
                price: 0,
                frequency: 'monthly',
                trial_days: 0,
                active: true,
                is_featured: false,
                features: '',
                credit_system_type: 'separated',
                shared_credits_amount: 0,
                credit_tier: 'standard',
                credit_limits: {},
                plan_ai_tools: { ...DEFAULT_TOOLS },
                plan_features: { ...DEFAULT_FEATURES },
            });

            Alpine.data('aiPlanWizard', () => ({
                isDemo: @json((bool) $app_is_demo),
                step: 'intro',
                stepLabels: {
                    basics: @json(__('Basics')),
                    credits: @json(__('Credits')),
                    features: @json(__('Features')),
                    review: @json(__('Review')),
                },
                creditTiers: [
                    { key: 'starter', label: @json(__('Starter')), multiplier: 0.5 },
                    { key: 'standard', label: @json(__('Standard')), multiplier: 1 },
                    { key: 'pro', label: @json(__('Pro')), multiplier: 2 },
                    { key: 'enterprise', label: @json(__('Enterprise')), multiplier: 5 },
                ],
                creditCategories: [
                    { key: 'word', label: @json(__('Word Models')) },
                    { key: 'image', label: @json(__('Image Models')) },
                    { key: 'video', label: @json(__('Video Models')) },
                    { key: 'audio', label: @json(__('Audio Models')) },
                    { key: 'presentation', label: @json(__('Presentation Models')) },
                ],
                quickPrompts: [
                    @json(__('Suggest a competitive price')),
                    @json(__('Write the feature list for me')),
                    @json(__('Build a $29 Pro plan')),
                ],
                form: freshForm(),
                created: null,
                presets: [],
                presetsLoaded: false,
                loadingPresets: false,
                validating: false,
                submitting: false,
                errors: {},
                messages: [],
                chatInput: '',
                streaming: false,

                get streamHasContent() {
                    const last = this.messages[this.messages.length - 1];

                    return !!(last && last.role === 'assistant' && last.content.length);
                },

                modal() {
                    return Alpine.$data(document.querySelector('#ai-plan-wizard-modal'));
                },

                blockedInDemo() {
                    if (! this.isDemo) return false;
                    toastr.info(@json(__('This feature is disabled in demo mode.')));

                    return true;
                },

                openWizard() {
                    this.modal().modalOpen = true;
                },

                visibleSteps() {
                    return ['basics', 'credits', 'features', 'review'];
                },

                stepIndex(step) {
                    return this.visibleSteps().indexOf(step);
                },

                async loadPresets() {
                    if (this.presetsLoaded || this.loadingPresets) return;
                    this.loadingPresets = true;
                    try {
                        const response = await fetch(ROUTES.presets, {
                            headers: { 'Accept': 'application/json' },
                        });
                        if (!response.ok) throw new Error('presets failed');
                        const data = await response.json();
                        this.presets = data.presets || [];
                        this.presetsLoaded = true;
                    } catch (error) {
                        this.presets = [];
                        this.presetsLoaded = true;
                    } finally {
                        this.loadingPresets = false;
                    }
                },

                applyPreset(preset) {
                    this.applyUpdates(preset);
                    this.step = 'basics';
                },

                startFromScratch() {
                    this.step = 'basics';
                },

                back() {
                    this.errors = {};
                    const steps = this.visibleSteps();
                    const index = steps.indexOf(this.step);
                    this.step = index <= 0 ? 'intro' : steps[index - 1];
                },

                payloadFor(step) {
                    const payload = { step };
                    (STEP_FIELDS[step] || []).forEach((field) => {
                        payload[field] = this.form[field];
                    });

                    return payload;
                },

                async next() {
                    if (this.validating) return;
                    this.validating = true;
                    this.errors = {};
                    try {
                        const response = await fetch(ROUTES.validateStep, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-CSRF-TOKEN': CSRF_TOKEN,
                            },
                            body: JSON.stringify(this.payloadFor(this.step)),
                        });
                        if (response.status === 422) {
                            const data = await response.json();
                            this.errors = data.errors || {};

                            return;
                        }
                        if (!response.ok) throw new Error('validation failed');
                        const steps = this.visibleSteps();
                        this.step = steps[steps.indexOf(this.step) + 1] || 'review';
                    } catch (error) {
                        toastr.error(@json(__('Something went wrong. Please try again.')));
                    } finally {
                        this.validating = false;
                    }
                },

                async submitPlan() {
                    if (this.blockedInDemo()) return;
                    if (this.submitting) return;
                    this.submitting = true;
                    this.errors = {};
                    const modal = this.modal();
                    modal.modalLocked = true;
                    try {
                        const response = await fetch(ROUTES.store, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'application/json',
                                'X-CSRF-TOKEN': CSRF_TOKEN,
                            },
                            body: JSON.stringify(this.form),
                        });
                        const data = await response.json().catch(() => ({}));
                        if (response.status === 422) {
                            this.errors = data.errors || {};
                            this.errors.general = data.message || @json(__('Please check the highlighted fields.'));

                            return;
                        }
                        if (!response.ok) {
                            this.errors.general = data.message || @json(__('Something went wrong. Please try again.'));

                            return;
                        }
                        toastr.success(data.message || @json(__('Plan created successfully.')));
                        this.created = {
                            message: data.message || @json(__('Plan created successfully.')),
                            editUrl: data.edit_url || null,
                        };
                        this.step = 'done';
                    } catch (error) {
                        this.errors.general = @json(__('Something went wrong. Please try again.'));
                    } finally {
                        modal.modalLocked = false;
                        this.submitting = false;
                    }
                },

                toggleAll(key) {
                    const map = this.form[key];
                    const enable = !Object.values(map).some(Boolean);
                    Object.keys(map).forEach((itemKey) => {
                        map[itemKey] = enable;
                    });
                },

                enabledCount(key) {
                    return Object.values(this.form[key]).filter(Boolean).length;
                },

                creditLimitsSummary() {
                    return this.creditCategories
                        .filter((category) => {
                            const value = this.form.credit_limits[category.key];

                            return value !== undefined && value !== null && value !== '' && !Number.isNaN(Number(value));
                        })
                        .map((category) => category.label + ': ' + Number(this.form.credit_limits[category.key]))
                        .join(' · ');
                },

                renderMarkdown(text) {
                    if (window.markdownit) {
                        this._md ??= window.markdownit({ html: false, linkify: true, breaks: true });

                        return this._md.render(text);
                    }

                    const escaped = document.createElement('span');
                    escaped.textContent = text;

                    return escaped.innerHTML;
                },

                displayContent(message) {
                    if (message.role !== 'assistant') return message.content;
                    let text = message.content.split(UPDATES_DELIMITER)[0];
                    for (let i = UPDATES_DELIMITER.length - 1; i > 0; i--) {
                        if (text.endsWith(UPDATES_DELIMITER.slice(0, i))) {
                            text = text.slice(0, -i);
                            break;
                        }
                    }

                    return text.trimEnd();
                },

                scrollChat() {
                    this.$nextTick(() => {
                        const container = document.getElementById('ai-wizard-chat-scroll');
                        if (container) container.scrollTop = container.scrollHeight;
                    });
                },

                async sendMessage() {
                    if (this.blockedInDemo()) return;
                    const text = this.chatInput.trim();
                    if (!text || this.streaming) return;
                    this.messages.push({ role: 'user', content: text });
                    this.chatInput = '';
                    this.streaming = true;
                    this.messages.push({ role: 'assistant', content: '' });
                    const assistantMessage = this.messages[this.messages.length - 1];
                    this.scrollChat();
                    try {
                        const history = this.messages
                            .slice(0, -1)
                            .slice(-20)
                            .map((message) => ({ role: message.role, content: this.displayContent(message) }));
                        const response = await fetch(ROUTES.chat, {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'Accept': 'text/event-stream',
                                'X-CSRF-TOKEN': CSRF_TOKEN,
                            },
                            body: JSON.stringify({ messages: history, draft: this.form }),
                        });
                        if (!response.ok || !response.body) {
                            const data = await response.json().catch(() => ({}));
                            assistantMessage.content = data.message || @json(__('Sorry, I could not reach the AI service. Please try again.'));

                            return;
                        }
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let buffer = '';
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            buffer += decoder.decode(value, { stream: true });
                            const events = buffer.split('\n\n');
                            buffer = events.pop();
                            events.forEach((event) => {
                                const line = event.trim();
                                if (!line.startsWith('data:')) return;
                                const payload = line.slice(5).trim();
                                if (payload === '[DONE]') return;
                                try {
                                    const parsed = JSON.parse(payload);
                                    assistantMessage.content += parsed.content || '';
                                } catch (error) {
                                    // Ignore malformed chunks
                                }
                            });
                            this.scrollChat();
                        }
                        assistantMessage.updated = this.applyUpdatesFrom(assistantMessage.content);
                        if (assistantMessage.updated && this.step === 'intro') {
                            this.step = 'basics';
                        }
                        if (assistantMessage.updated && !this.displayContent(assistantMessage).trim()) {
                            assistantMessage.content = @json(__('Done — I\'ve updated the plan draft.')) + assistantMessage.content;
                        }
                        if (!assistantMessage.content.trim()) {
                            assistantMessage.content = @json(__('Sorry, I could not reach the AI service. Please try again.'));
                        }
                    } catch (error) {
                        assistantMessage.content = @json(__('Sorry, I could not reach the AI service. Please try again.'));
                    } finally {
                        this.streaming = false;
                        this.scrollChat();
                    }
                },

                applyUpdatesFrom(content) {
                    const index = content.indexOf(UPDATES_DELIMITER);
                    if (index === -1) return false;
                    let raw = content.slice(index + UPDATES_DELIMITER.length).trim();
                    raw = raw.replace(/```(?:json)?/g, '').trim();
                    const start = raw.indexOf('{');
                    const end = raw.lastIndexOf('}');
                    if (start === -1 || end <= start) return false;
                    try {
                        return this.applyUpdates(JSON.parse(raw.slice(start, end + 1)));
                    } catch (error) {
                        return false;
                    }
                },

                applyUpdates(updates) {
                    if (!updates || typeof updates !== 'object') return false;
                    let applied = false;
                    ['plan_ai_tools', 'plan_features'].forEach((mapField) => {
                        const only = updates[mapField + '_only'];
                        if (!Array.isArray(only)) return;
                        Object.keys(this.form[mapField]).forEach((itemKey) => {
                            this.form[mapField][itemKey] = only.includes(itemKey);
                        });
                        applied = true;
                    });
                    ALLOWED_UPDATE_FIELDS.forEach((field) => {
                        if (!(field in updates)) return;
                        const value = updates[field];
                        if (field === 'credit_limits') {
                            if (value && typeof value === 'object') {
                                Object.keys(value).forEach((category) => {
                                    if (CREDIT_CATEGORIES.includes(category) && !Number.isNaN(Number(value[category]))) {
                                        this.form.credit_limits[category] = Math.max(0, Number(value[category]));
                                        applied = true;
                                    }
                                });
                            }

                            return;
                        }
                        if (field === 'plan_ai_tools' || field === 'plan_features') {
                            if (value && typeof value === 'object') {
                                Object.keys(value).forEach((itemKey) => {
                                    if (itemKey in this.form[field]) {
                                        this.form[field][itemKey] = !!value[itemKey];
                                        applied = true;
                                    }
                                });
                            }

                            return;
                        }
                        if (['active', 'is_featured'].includes(field)) {
                            this.form[field] = !!value;
                            applied = true;

                            return;
                        }
                        this.form[field] = value;
                        applied = true;
                    });

                    return applied;
                },
            }));
        });
    </script>
@endpush
