<aside
    class="lqd-adv-img-editor-sidebar group/sidebar fixed bottom-0 end-0 top-0 w-[--sidebar-w] bg-background pt-[--header-h] shadow-[0_4px_44px_rgba(0,0,0,0.06)] transition-all motion-duration-[0.35s] group-[&.active]/editor:motion-preset-slide-left-md group-[&.sidebar-collapsed]/editor:translate-x-[calc(100%-40px)] group-[&.active]/editor:motion-opacity-in-[0%]"
    x-ref="imageEditorSidebar"
>
    <x-button
        class="size-[38px] absolute start-0 top-24 z-10 -translate-x-1/2 bg-background motion-duration-[0.35s] motion-delay-200 group-[&.active]/editor:motion-preset-slide-left-md"
        variant="outline"
        size="none"
        hover-variant="primary"
        @click.prevent="switchSidebarCollapsed()"
        ::class="{ 'rotate-180': sidebarCollapsed }"
    >
        <x-tabler-chevron-right class="size-4" />
    </x-button>

    <form
        class="lqd-adv-img-editor-sidebar-inner h-full overflow-y-auto px-7 pb-9 pt-7 transition-all duration-300 motion-duration-[0.45s] motion-delay-150 group-[&.active]/editor:motion-preset-slide-left-sm group-[&.sidebar-collapsed]/editor:translate-x-2 group-[&.sidebar-collapsed]/editor:opacity-0"
        action="{{ route('dashboard.user.advanced-image.editor') }}"
        method="POST"
        enctype="multipart/form-data"
        @submit.prevent="submitEditorForm"
    >
        @csrf
        <div class="flex min-h-full grow flex-col gap-y-7">
            <div
                class="group/dropdown relative"
                x-data='{
                    dropdownOpen: false,
                }'
                :class="{ open: dropdownOpen }"
                @click.outside="dropdownOpen = false"
            >
                <button
                    class="group/dropdown-trigger flex w-full items-center gap-3 rounded-lg bg-heading-foreground/[4%] px-3.5 py-[5px] text-sm font-semibold text-heading-foreground"
                    type="button"
                    @click="dropdownOpen = !dropdownOpen"
                >
                    <span
                        class="size-11 relative inline-grid shrink-0 place-items-center rounded-full border border-heading-foreground/10 transition-all group-hover/dropdown:bg-heading-foreground group-hover/dropdown:text-heading-background group-hover/dropdown:shadow-xl group-hover/dropdown:shadow-black/10"
                        x-html="tools.find(a => a.action === selectedTool)?.icon"
                    >
                    </span>
                    <span x-text="tools.find(a => a.action === selectedTool)?.title"></span>

                    <x-tabler-chevron-down class="size-4 ms-auto" />
                </button>
                <ul
                    class="invisible absolute inset-x-0 top-full z-2 mt-1.5 flex max-h-[300px] w-full origin-top -translate-y-1 scale-90 flex-col overflow-y-auto rounded-lg border bg-background p-2 opacity-0 shadow-xl shadow-black/10 transition-all group-[&.open]/dropdown:visible group-[&.open]/dropdown:translate-y-0 group-[&.open]/dropdown:scale-100 group-[&.open]/dropdown:opacity-100">
                    <template
                        x-for="tool in tools"
                        :key="tool.action"
                    >
                        <li class="border-b last:border-b-0">
                            <button
                                class="flex w-full items-center gap-3 rounded px-4 py-1 text-xs font-semibold text-heading-foreground transition-all hover:bg-heading-foreground/5"
                                type="button"
                                @click="selectedTool = tool.action; switchToolsCat({toolKey: tool.action}); dropdownOpen = false"
                                :class="{ 'text-primary bg-primary/[7%]': selectedTool === tool.action }"
                            >
                                <span
                                    class="size-11 inline-grid place-content-center rounded-full border border-heading-foreground/10 [&_svg]:h-auto [&_svg]:w-full"
                                    x-html="tool.icon"
                                ></span>
                                <span x-text="tool.title"></span>
                            </button>
                        </li>
                    </template>
                </ul>
            </div>

			<template x-if="! ['remove_text', 'cleanup', 'remove_background', 'upscale', 'merge_face'].includes(selectedTool)">
				<x-forms.input
					class:label-txt="flex items-center gap-2 text-label text-2xs text-heading-foreground/80"
					type="textarea"
					label="{{ __('Prompt') }}"
					placeholder="{{ __('Describe your idea or select a pre-defined prompt') }}"
					name="description"
					rows="8"
					size="lg"
					x-model="prompt"
					x-ref="promptInput"
				>
					<x-slot:label>
						{{ __('Prompt') }}
						<x-button
							class="size-6 shrink-0 justify-center self-center text-center hover:translate-y-0 hover:scale-105 hover:shadow-none"
							title="{{ __('Browse pre-defined prompts') }}"
							variant="none"
							size="none"
							@click.prevent="setRandomPrompt()"
						>
							<svg
								class="size-5"
								width="19"
								height="20"
								viewBox="0 0 19 20"
								fill="currentColor"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path
									d="M1 16.0212C1.1795 15.9071 1.37075 15.8109 1.57375 15.7327C1.77675 15.6546 1.99575 15.6155 2.23075 15.6155H3.3845V3H2.23075C1.88208 3 1.58975 3.1215 1.35375 3.3645C1.11792 3.60733 1 3.89608 1 4.23075V16.0212ZM2.23075 20C1.61108 20 1.08442 19.7868 0.65075 19.3605C0.216917 18.9343 0 18.4167 0 17.8078V4.23075C0 3.61108 0.216917 3.08442 0.65075 2.65075C1.08442 2.21692 1.61108 2 2.23075 2H9.5V3H4.3845V15.6155H11.6155V11.5H12.6155V16.6155H2.23075C1.89608 16.6155 1.60733 16.7294 1.3645 16.9573C1.1215 17.1851 1 17.4674 1 17.8043C1 18.1411 1.1215 18.4246 1.3645 18.6548C1.60733 18.8849 1.89608 19 2.23075 19H15V10.5H16V20H2.23075ZM13.5 10.5C13.5 9.106 13.9848 7.92417 14.9545 6.9545C15.9242 5.98483 17.106 5.5 18.5 5.5C17.106 5.5 15.9242 5.01517 14.9545 4.0455C13.9848 3.07583 13.5 1.894 13.5 0.5C13.5 1.894 13.0152 3.07583 12.0455 4.0455C11.0758 5.01517 9.894 5.5 8.5 5.5C9.894 5.5 11.0758 5.98483 12.0455 6.9545C13.0152 7.92417 13.5 9.106 13.5 10.5Z"
								/>
							</svg>
						</x-button>
					</x-slot:label>
				</x-forms.input>
			</template>

			<input hidden x-model="aiModel" value="clipdrop" name="ai_model">



			<template x-if="['reimagine'].includes(selectedTool)">
				<x-forms.input
					class:label="text-heading-foreground/80"
					size="lg"
					name="imagination"
					label="{{ __('Imagination') }}"
					type="select"
				>
					<option value="wild">Wild</option>
					<option value="subtle">Subtle</option>
					<option value="vivid">Vivid</option>
				</x-forms.input>
			</template>

			<template x-if="['style_transfer'].includes(selectedTool)">
				<x-forms.input type="file" name="reference_image" label="Reference image" />
			</template>

			<template x-if="[ 'merge_face'].includes(selectedTool)">
				<x-forms.input type="file" name="face_image" label="Face image" />
			</template>

			<template x-if="[ 'image_relight'].includes(selectedTool)">
				<x-forms.input type="file" name="image_relight" label="Transfer light from reference image" />
			</template>

			<template x-if="['image_relight'].includes(selectedTool)">
				<x-forms.input
					class:label="text-heading-foreground/80"
					size="lg"
					name="style"
					label="{{ __('Style') }}"
					type="select"
				>
					<option value="standard">Standard</option>
					<option value="darker_but_realistic">Darker but realistic</option>
					<option value="clean">Clean</option>
					<option value="smooth">Smooth</option>
					<option value="brighter">Brighter</option>
					<option value="contrasted_n_hdr">Contrasted n hdr</option>
					<option value="just_composition">Just composition</option>
				</x-forms.input>
			</template>

			<template x-if="['reimagine'].includes(selectedTool)">
				<x-forms.input
					class:label="text-heading-foreground/80"
					size="lg"
					name="aspect_ratio"
					label="{{ __('Aspect ratio') }}"
					type="select"
				>
					<option value="original">Original</option>
					<option value="square_1_1">Square 1x1</option>
					<option value="classic_4_3">Classic_4x3</option>
					<option value="traditional_3_4">Traditional 3x4</option>
					<option value="widescreen_16_9">widescreen_16_9</option>
					<option value="social_story_9_16">social_story_9_16</option>
					<option value="standard_3_2">standard_3_2</option>
					<option value="portrait_2_3">portrait_2_3</option>
					<option value="horizontal_2_1">horizontal_2_1</option>
					<option value="vertical_1_2">vertical_1_2</option>
					<option value="social_post_4_5">social_post_4_5</option>
				</x-forms.input>
			</template>
{{--			<div--}}
{{--				x-show="selectedTool === 'reimagine'"--}}
{{--			>--}}
{{--				<p class="text-2xs/5 opacity-60">--}}
{{--					{{__("Please upload an image with dimensions of 1024 x 1024.")}}--}}
{{--				</p>--}}
{{--			</div>--}}

            <div
                class="space-y-7"
                x-show="selectedTool === 'uncrop'"
                x-collapse
            >
                <x-forms.input
                    size="lg"
                    name="extend_left"
                    label="{{ __('Extend Left') }}"
                    type="number"
                    step="10"
                    stepper
                />
                <x-forms.input
                    size="lg"
                    name="extend_up"
                    label="{{ __('Extend Up') }}"
                    type="number"
                    step="10"
                    stepper
                />
                <x-forms.input
                    size="lg"
                    name="extend_right"
                    label="{{ __('Extend Right') }}"
                    type="number"
                    step="10"
                    stepper
                />
                <x-forms.input
                    size="lg"
                    name="extend_down"
                    label="{{ __('Extend Down') }}"
                    type="number"
                    step="10"
                    stepper
                />
            </div>

{{--            <div--}}
{{--                x-show="selectedTool === 'upscale'"--}}
{{--                x-collapse--}}
{{--            >--}}
{{--				--}}
{{--				<x-forms.input--}}
{{--					class:label="text-heading-foreground/80"--}}
{{--					size="lg"--}}
{{--					name="scale_factor"--}}
{{--					label="{{ __('Scale factor') }}"--}}
{{--					type="select"--}}
{{--				>--}}
{{--					<option value="2x">2x</option>--}}
{{--					<option value="4x">4x</option>--}}
{{--		--}}
{{--				</x-forms.input>--}}
{{--            </div>--}}

            <input
                type="hidden"
                name="selected_tool"
                :value="selectedTool"
            />
            <input
                class="hidden"
                type="file"
                name="uploaded_image"
                accept="image/png, image/jpeg"
                x-ref="uploadedImageInput"
            >
            <input
                class="hidden"
                name="mask_file"
                type="file"
                accept="image/png"
                x-ref="maskFileInput"
            />
            <input
                class="hidden"
                name="sketch_file"
                type="file"
                accept="image/png"
                x-ref="sketchFileInput"
            />

            <x-button
                class="mt-auto w-full"
                size="lg"
                type="submit"
                style="background: linear-gradient(90deg, #8364FF 2.78%, #5EACFF 32.43%, #4FACBE 63.99%, #BF58FF 100%);"
            >
                @lang('Generate')
            </x-button>
        </div>
    </form>
</aside>
