@php
    $predefined_prompts = [
        __('Delicious pizza with all the toppings'),
        __('Cityscape at sunset in retro vector illustration'),
        __('Photo of an extremely cute alien fish swimming an alien habitable underwater planet'),
        __('Painting of a flower vase on a kitchen table with a window in the backdrop'),
        __('A super detailed infographic of a working time machine 8k'),
        __('Illustration of a cat sitting on a couch in a living room with a coffee mug in its hand'),
        __('Freeform ferrofluids, beautiful dark chaos'),
        __('Hedgehog smelling a flower'),
    ];

    $primary_tool_keys = ['uncrop', 'reimagine', 'remove_background', 'cleanup', 'upscale'];

@endphp

@extends('panel.layout.app', [
    'disable_tblr' => true,
    'disable_header' => true,
    'disable_navbar' => true,
    'disable_default_sidebar' => true,
    'disable_titlebar' => true,
    'layout_wide' => true,
])
@section('title', __('Advanced Image Editor'))
@section('titlebar_actions', '')

@section('content')
    <div
        class="pointer-events-none absolute inset-x-0 top-0 z-0 overflow-hidden opacity-30 dark:hidden"
        aria-hidden="true"
    >
        <img
            class="w-full"
            src="{{ custom_theme_url('assets/img/advanced-image/image-editor-bg.jpg') }}"
            alt="Background image"
        >
    </div>

    <div
        class="lqd-adv-img-editor relative z-1 pt-[--header-h] [--header-h:60px] [--sidebar-w:370px]"
        x-data='advancedImageEditor({
            predefinedPrompts: @json($predefined_prompts),
			tools: @json($tools),
            primaryToolKeys: @json($primary_tool_keys),
        })'
        @keyup.escape.window="!modalShow && switchView('<')"
    >
        @include('advanced-image::shared-components.top-navbar')

        @include('advanced-image::home.home', ['images' => $userOpenai->take(5), 'tools' => $tools])

        @include('advanced-image::editor.editor', ['tools' => $tools, 'primary_tool_keys' => $primary_tool_keys])

        @include('advanced-image::gallery.gallery', ['images' => $userOpenai])

        @include('advanced-image::shared-components.image-modal')
    </div>


@endsection

@push('script')
	<script>

		const AIModelsforTool = @json(\Illuminate\Support\Arr::pluck($tools, 'model', 'action'));

		async function fetchImageStatus() {
			try {
				const response = await fetch('/dashboard/user/openai/generator/check/status');
				const data = await response.json();
				if (data.data) {
					data.data.forEach(item => updateImage(item));
				}
			} catch (error) {
				console.error('Error fetching image status:', error);
			}
		}

		function updateImage(item) {
			const wrapperEl = document.querySelector(`[data-id="${item.id}"]`);
			if (!wrapperEl) return;

			const imgElement = wrapperEl.querySelector(`#img-${item.response}-${item.id}`);
			const imgElementPayloadId = wrapperEl.querySelector(`#img-${item.response}-${item.id}-payload`);
			const imgElementDownload = wrapperEl.querySelector(`#img-${item.response}-${item.id}-download`);

			if (imgElement) {
				imgElement.src = item.img;
				imgElement.alt = item.title;
			}

			if (imgElementDownload) {
				imgElementDownload.href = item.img;
				imgElementDownload.target = '_blank';
			}

			wrapperEl.setAttribute('data-payload', JSON.stringify(item));
			refreshFsLightbox();
		}

		document.addEventListener('DOMContentLoaded', () => {
			setInterval(fetchImageStatus, 5000);
		});
	</script>
@endpush

