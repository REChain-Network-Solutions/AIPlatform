@php
	use App\Domains\Entity\Enums\EntityEnum;

	$currentUrl = url()->current();
	$previousUrl = url()->previous();

	$is_chat_pro =
		\App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro') &&
		(route('dashboard.user.openai.chat.pro.index') === $currentUrl ||
			route('chat.pro') === $currentUrl ||
			route('dashboard.user.openai.chat.pro.index') === $previousUrl ||
			route('chat.pro') === $previousUrl);

	$isOtherCategories = isset($category) && ($category->slug == 'ai_vision' || $category->slug == 'ai_pdf' || $category->slug == 'ai_chat_image');
@endphp

<div
	class="lqd-chat-head sticky -top-px z-30 flex min-h-20 items-center justify-between gap-2 rounded-se-[inherit] border-b bg-background/80 px-5 py-3 backdrop-blur-lg backdrop-saturate-150 max-md:bg-background/95 max-md:px-4">
	<div class="flex flex-col items-start justify-center text-sm">
		@include('panel.user.openai_chat.components.chat_category_dropdown')
	</div>

	<div class="flex grow items-center justify-end gap-4">
		<div class="flex gap-2">
			@if (!$isOtherCategories && $is_chat_pro)
				<x-button
					class="lqd-mobile-model-modal-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
					variant="none"
					size="none"
					x-data=""
					@click.prevent="document.querySelector('.select-ai-model-modal') && Alpine.$data(document.querySelector('.select-ai-model-modal')).toggleModal()"
				>
					<x-tabler-brand-openai class="size-5"/>
				</x-button>
			@endif

			@includeFirst(['chat-share::share-button-include', 'panel.user.openai_chat.includes.share-button-include', 'vendor.empty'])

			@if (view()->hasSection('chat_head_actions'))
				@yield('chat_head_actions')
			@else
				@php
					$realtimeHiddenIn = ['ai_pdf', 'ai_vision', 'ai_chat_image'];
				@endphp
				@auth
					<div x-data="realtimeToggle">
						<x-forms.input
							class="max-md:hidden"
							id="realtime"
							x-ref="realtimeCheckbox"
							container-class="{{ in_array($category->slug, $realtimeHiddenIn, true) ? 'hidden' : 'flex' }} max-md:size-8 max-md:inline-flex max-md:items-center max-md:justify-center max-md:overflow-hidden max-md:shadow-md max-md:rounded-full max-md:shrink-0 max-md:[&_.lqd-input-label-txt]:hidden"
							label="{{ __('Real-Time Data') }}"
							type="checkbox"
							name="realtime"
							@change="handleRealtimeChange($event)"
							switcher
						>
							<span
								class="inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-background indent-0 text-heading-foreground transition-colors peer-checked:bg-primary peer-checked:text-primary-foreground md:hidden"
							>
								<x-tabler-world-download
									class="size-5"
									stroke-width="1.5"
								/>
							</span>
						</x-forms.input>
					</div>
				@else
					<x-forms.input
						class="max-md:hidden"
						id="realtime"
						container-class="{{ in_array($category->slug, $realtimeHiddenIn, true) ? 'hidden' : 'flex' }} max-md:size-8 max-md:inline-flex max-md:items-center max-md:justify-center max-md:overflow-hidden max-md:shadow-md max-md:rounded-full max-md:shrink-0 max-md:[&_.lqd-input-label-txt]:hidden"
						label="{{ __('Real-Time Data') }}"
						type="checkbox"
						name="realtime"
						onchange="toastr.warning('{{ __('Login to use Real-Time search') }}'); document.querySelector('#realtime').checked = false; return false;"
						switcher
					>
						<span
							class="inline-flex size-8 shrink-0 items-center justify-center rounded-full bg-background indent-0 text-heading-foreground transition-colors peer-checked:bg-primary peer-checked:text-primary-foreground md:hidden"
						>
							<x-tabler-world-download
								class="size-5"
								stroke-width="1.5"
							/>
						</span>
					</x-forms.input>
				@endauth
			@endif

			<div
				class="group relative inline-flex flex-row items-center justify-center self-center max-md:-order-1"
				id="show_export_btns"
			>
				<button class="max-md:inline-flex max-md:size-8 max-md:items-center max-md:justify-center max-md:rounded-full max-md:shadow-md">
					<x-tabler-clipboard-copy
						class="size-6"
						stroke-width="1.5"
					/>
				</button>
				<div
					class="invisible absolute -end-4 bottom-full flex translate-y-2 scale-95 flex-row items-center justify-center rounded-lg bg-primary text-primary-foreground opacity-0 transition-all group-focus-within:visible group-focus-within:translate-y-0 group-focus-within:scale-100 group-focus-within:opacity-100 group-hover:visible group-hover:translate-y-0 group-hover:scale-100 group-hover:opacity-100"
					id="export_btns"
				>
					<button
						class="chat-download border-none px-3 py-1 text-3xs font-medium"
						id="export_pdf"
						data-doc-type="pdf"
					>
						{{ __('PDF') }}
					</button>
					<button
						class="chat-download border-x border-x-primary-foreground/20 px-2.5 py-1 text-3xs font-medium"
						id="export_word"
						data-doc-type="doc"
					>
						{{ __('Word') }}
					</button>
					<button
						class="chat-download px-3 py-1 text-3xs font-medium"
						id="export_txt"
						data-doc-type="txt"
					>
						{{ __('Txt') }}
					</button>
				</div>
			</div>

			@if (view()->hasSection('chat_sidebar_actions'))
				@yield('chat_sidebar_actions')
			@else
				@if (isset($category) && $category->slug == 'ai_pdf')
					{{-- #selectDocInput is present in chat_sidebar component. no need to duplicate it here --}}
					<x-button
						class="lqd-upload-doc-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
						variant="none"
						size="none"
						href="javascript:void(0);"
						onclick="return $('#selectDocInput').click();"
					>
						<x-tabler-plus class="size-5"/>
						<span class="sr-only">
                            {{ __('Upload Document') }}
                        </span>
					</x-button>
				@else
					<x-button
						class="lqd-new-chat-trigger group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
						variant="none"
						size="none"
						href="javascript:void(0);"
						onclick="{!! $disable_actions
                            ? 'return toastr.info(\'{{ __('This feature is disabled in Demo version.') }}\')'
                            : (auth()->check()
                                ? 'return startNewChat(\'{{ $category->id }}\', \'{{ LaravelLocalization::getCurrentLocale() }}\', \'chatpro\')'
                                : 'return window.location.reload();') !!}"
					>
						<x-tabler-plus class="size-5"/>
						<span class="sr-only">
                            {{ __('New Conversation') }}
                        </span>
					</x-button>
				@endif

				<div class="lqd-chat-mobile-sidebar-trigger self-center">
					<button
						class="group size-8 shrink-0 grid-flow-row place-items-center rounded-full shadow-md max-md:grid md:hidden"
						:class="{ 'active': mobileSidebarShow }"
						@click.prevent="toggleMobileSidebar()"
						type="button"
					>
						<x-tabler-dots
							class="col-start-1 row-start-1 size-5 transition-all group-[&.active]:rotate-45 group-[&.active]:scale-75 group-[&.active]:opacity-0"/>
						<x-tabler-x
							class="col-start-1 row-start-1 size-4 -rotate-45 opacity-0 transition-all group-[&.active]:rotate-0 group-[&.active]:!opacity-100"/>
					</button>
				</div>
			@endif
		</div>
	</div>
</div>

@pushOnce('script')
	<script>
		document.addEventListener('alpine:init', () => {
			Alpine.data('realtimeToggle', () => ({
				isRealtimeOpenAI: "{{ setting('default_realtime') === 'openai' }}" === "1",
				defaultOpenAIRealtimeModel: "{{ setting('openai_realtime_model', EntityEnum::GPT_4_O_SEARCH_PREVIEW->value) }}",
				previousModel: null,
				init() {},
				handleRealtimeChange(event) {
					const isChecked = event.target.checked;

					if (isChecked) {
						if (this.isRealtimeOpenAI) {
							// Store current model before switching
							const localStorageKey = 'selectedChatModels';
							const storedModels = localStorage.getItem(localStorageKey);

							if (storedModels) {
								const models = JSON.parse(storedModels);
								this.previousModel = models[0]?.value;
							}

							this.switchToRealtimeModel();
							toastr.success('{{ __("Real-Time data activated") }}');
						} else {
							toastr.success('{{ __("Real-Time data activated") }}');
						}
					} else {
						if (this.isRealtimeOpenAI && this.previousModel) {
							this.revertToPreviousModel();
						}
						toastr.warning('{{ __("Real-Time data deactivated") }}');
					}
				},
				switchToRealtimeModel() {
					// Dispatch event to change model
					document.dispatchEvent(new CustomEvent('chat-model-change', {
						detail: { model: this.defaultOpenAIRealtimeModel }
					}));
				},
				revertToPreviousModel() {
					if (this.previousModel) {
						document.dispatchEvent(new CustomEvent('chat-model-change', {
							detail: { model: this.previousModel }
						}));
						this.previousModel = null;
					}
				}
			}));
		});
	</script>
@endPushOnce
