<div class="col-span-full grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-4 lg:py-2.5">
	<div class="col-span-full flex items-center gap-7">
		<h3 class="m-0">
			@lang('Templates')
		</h3>

		<span class="inline-flex h-px grow bg-border"></span>

		<x-button
			class="!text-2xs"
			variant="link"
			href="{{ route('dashboard.user.advanced-image.index') }}"
		>
			{{ __('View All') }}
			<x-tabler-circle-chevron-right
				class="size-5"
				stroke-width="1.5"
			/>
		</x-button>
	</div>

	@php
		$templates = [
			[
				'image' => 'assets/img/advanced-image/templates/templates-facebook-cover.svg',
				'title' => 'Facebook Cover',
				'aspect' =>
					'Create a visually striking Facebook cover image with a resolution of 820x312 pixels, emphasizing a wide landscape layout. Suitable for personal or business pages, with space for profile picture overlay on the left.',
				'cover' => 'Horizontal',
			],
			[
				'image' => 'assets/img/advanced-image/templates/templates-youtube-thumbnail.svg',
				'title' => 'Youtube Thumbnail',
				'aspect' =>
					'Design an eye-catching YouTube thumbnail with a resolution of 1280x720 pixels, highlighting key elements of the video. Use bold fonts, vibrant colors, and clear focal points to grab viewer attention.',
				'cover' => 'Horizontal',
			],
			[
				'image' => 'assets/img/advanced-image/templates/templates-advertisment.svg',
				'title' => 'Advertisment',
				'aspect' =>
					'Create a visually appealing square advertisement image with a resolution of 1080x1080 pixels, ideal for social media platforms like Instagram and Facebook. Ensure the design is vibrant, with a clear call-to-action.',
				'cover' => 'Square',
			],
			[
				'image' => 'assets/img/advanced-image/templates/templates-reels-tiktok-youtube-short.svg',
				'title' => 'Reels, TikTok, Youtube Short',
				'aspect' =>
					'Generate an engaging vertical video cover image with a resolution of 1080x1920 pixels, suitable for platforms like Reels, TikTok, and YouTube Shorts. The design should be visually dynamic and capture the viewer\'s attention within a short time frame.',
				'cover' => 'Vertical',
			],
			[
				'image' => 'assets/img/advanced-image/templates/templates-social-media-post.svg',
				'title' => 'Social Media Post',
				'aspect' =>
					'Create a square social media post image with a resolution of 1080x1080 pixels. This design should be versatile, suitable for multiple platforms, and include space for text or branding elements.',
				'cover' => 'Square',
			],
		];
	@endphp

	<div class="col-span-full grid grid-cols-1 gap-5 sm:grid-cols-2 md:grid-cols-3 md:gap-x-6 lg:grid-cols-5 lg:gap-x-11">
		@foreach ($templates as $template)
			<div class="lqd-adv-editor-templates-grid-item group/item relative">
				<div class="relative mb-4 grid aspect-[1/0.85] w-full items-end overflow-hidden rounded-lg border transition-all group-hover/item:-translate-y-1">
					<img
						class="h-auto w-full hue-rotate-[150deg] transition-all group-hover/item:scale-105"
						src="{{ custom_theme_url($template['image']) }}"
						alt="{{ $template['title'] }}"
					/>
				</div>
				<h5 class="mb-0 text-xs">
					{{ $template['title'] }}
				</h5>
				<p class="m-0 opacity-70">
					{{ $template['cover'] }}
				</p>

				<a
					class="template-link absolute inset-0"
					data-template-title="{{ $template['title'] }}"
					data-template-aspect="{{ $template['aspect'] }}"
					href="#"
					@click.prevent='selectedTemplate = "{{ $template['title'] }}"; selectedTemplateDescription = "{{ $template['aspect'] }}"; selectedPromptDescription = ""; window.scrollTo({ top: 0, behavior: "smooth" });'
				>
					<span class="sr-only">View Template</span>
				</a>
			</div>
		@endforeach
	</div>
</div>
