@php
	$image_styles = [
		'' => 'None',
		'3d_render' => '3D Render',
		'anime' => 'Anime',
		'ballpoint_pen' => 'Ballpoint Pen Drawing',
		'bauhaus' => 'Bauhaus',
		'cartoon' => 'Cartoon',
		'clay' => 'Clay',
		'contemporary' => 'Contemporary',
		'cubism' => 'Cubism',
		'cyberpunk' => 'Cyberpunk',
		'glitchcore' => 'Glitchcore',
		'impressionism' => 'Impressionism',
		'isometric' => 'Isometric',
		'line' => 'Line Art',
		'low_poly' => 'Low Poly',
		'minimalism' => 'Minimalism',
		'modern' => 'Modern',
		'origami' => 'Origami',
		'pencil' => 'Pencil Drawing',
		'pixel' => 'Pixel',
		'pointillism' => 'Pointillism',
		'pop' => 'Pop',
		'realistic' => 'Realistic',
		'renaissance' => 'Renaissance',
		'retro' => 'Retro',
		'steampunk' => 'Steampunk',
		'sticker' => 'Sticker',
		'ukiyo' => 'Ukiyo',
		'vaporwave' => 'Vaporwave',
		'vector' => 'Vector',
		'watercolor' => 'Watercolor',
	];
	$prompt_filters = [
		'all' => __('All'),
		'favorite' => __('Favorite'),
	];
@endphp

@extends('panel.layout.app', [
    'disable_tblr' => true,
    'disable_header' => true,
    'disable_navbar' => true,
    'disable_default_sidebar' => true,
    'disable_titlebar' => true,
    'layout_wide' => true,
])
@section('title', __('AI Real Time Image'))
@section('titlebar_subtitle')
	{{ __('View and manage external image') }}
@endsection
@section('titlebar_actions', '')

@push('css')
	<style>
		@media (min-width: 992px) {
			.lqd-page-content-wrap {
				min-height: 100vh;
			}

			.lqd-page-content-container {
				display: flex;
				flex-direction: column;
			}
		}
	</style>
@endpush

@section('content')
	<div
		class="lqd-realtime-image-decor-img pointer-events-none absolute inset-x-0 top-0 z-0 overflow-hidden opacity-30 dark:hidden"
		aria-hidden="true"
	>
		<img
			class="w-full"
			src="{{ custom_theme_url('assets/img/advanced-image/image-editor-bg.jpg') }}"
			alt="Background image"
		>
	</div>
	<div
		class="lqd-realtime-image relative z-1 pt-[--header-h] [--header-h:60px] [--sidebar-w:370px] lg:grow"
		x-data="liquidRealtimeImage"
		@keyup.escape.window="!modalShow && switchView('<')"
	>
		@include('ai-realtime-image::shared-components.top-navbar')

		@include('ai-realtime-image::home.home', ['image_styles' => $image_styles, 'prompt_filters' => $prompt_filters])

		@include('ai-realtime-image::gallery.gallery')

		@include('ai-realtime-image::shared-components.image-modal')
	</div>
@endsection

@push('script')
	<script>
		(() => {
			document.addEventListener('alpine:init', () => {
				Alpine.data('liquidRealtimeImage', () => ({
					prevViews: [],
					currentView: 'home',
					newImages: [],
					prompt: '',
					debouncedPrompt: '',
					lastImage: null,
					busy: false,
					requestSent: false,
					imageStyle: '{{ array_key_first($image_styles) }}',
					modalShow: false,
					activeModal: null,
					activeModalId: null,
					activeModalIdPrefix: null,
					promptLibraryShow: false,
					promptFilter: 'all',
					searchPromptStr: '',
					imageStyles: @json($image_styles),

					init() {
						this.changePrompt = Alpine.debounce(this.changePrompt.bind(this), 350);
					},

					switchView(view) {
						if (view === '<') {
							this.currentView = this.prevViews.pop() || 'home';
							return;
						}

						this.prevViews.push(this.currentView);
						this.currentView = view || 'home';
					},

					setActiveModal(data, idPrefix = 'modal') {
						this.activeModal = data;
						this.activeModalId = data.id;
						this.activeModalIdPrefix = idPrefix;
					},
					prevImageModal() {
						const currentEl = document.querySelector(`.image-result[data-id='${this.activeModalId}'][data-id-prefix=${this.activeModalIdPrefix}]`);
						const prevEl = currentEl?.previousElementSibling;
						if (!prevEl || !prevEl?.classList?.contains('image-result')) return;
						const data = JSON.parse(prevEl.getAttribute('data-payload') || {});
						this.setActiveModal(data, currentEl.getAttribute('data-id-prefix'));
					},
					nextImageModal() {
						const currentEl = document.querySelector(`.image-result[data-id='${this.activeModalId}'][data-id-prefix=${this.activeModalIdPrefix}]`);
						const nextEl = currentEl?.nextElementSibling;
						if (!nextEl || !nextEl?.classList?.contains('image-result')) return;
						const data = JSON.parse(nextEl.getAttribute('data-payload') || {});
						this.setActiveModal(data, currentEl.getAttribute('data-id-prefix'));
					},

					togglePromptLibraryShow() {
						this.promptLibraryShow = !this.promptLibraryShow
					},
					changePromptFilter(filter) {
						filter !== this.promptFilter && (this.promptFilter = filter)
					},
					setSearchPromptStr(str) {
						this.searchPromptStr = str.trim().toLowerCase()
					},
					setPrompt(prompt) {
						this.prompt = prompt;
					},
					focusOnPrompt() {
						this.$nextTick(() => {
							this.$refs.prompt.focus();
							this.onPromptInput();
						})
					},

					onPromptInput() {
						if (this.prompt.trim().length === 0 || this.debouncedPrompt.trim() === this.prompt.trim()) return;

						this.busy = true;

						this.changePrompt();
					},

					async changePrompt() {
						let formData = new FormData();

						formData.append('prompt', this.prompt);
						formData.append('style', this.imageStyle);

						this.busy = true;
						this.requestSent = true;

						this.debouncedPrompt = this.prompt;

						try {
							const response = await fetch('{{ route('dashboard.user.ai-realtime-image.store') }}', {
								method: 'POST',
								headers: {
									'X-CSRF-TOKEN': '{{ csrf_token() }}',
									'Accept': 'application/json',
								},
								body: formData,
							});

							const data = await response.json();

							if (data && data.status === '{{ \App\Extensions\AIRealtimeImage\System\Enums\Status::success->value }}') {
								this.newImages.unshift({
									...data.data || {},
									payload: data.payload,
									formatted_date: data.formatted_date
								});
								toastr.remove();
								toastr.success('{{ __('Image generated successfully') }}');
							} else {
								toastr.remove();
								if(data.message) {
									toastr.error(data.message);
								} else {
									toastr.error('{{ __('Error occurred while generating the image') }}');
								}
								console.log('Data Error:', data);
							}
						} catch (error) {
							toastr.remove();
							toastr.error('{{ __('Error occurred while generating the image') }}');
							console.log('Error:', error);
						} finally {
							this.busy = false;
							this.requestSent = false;
						}
					},
				}));
			});
		})();
	</script>
@endpush
