@php
    $container_base_class = 'lqd-input-container relative';
    $input_base_class = 'lqd-input block peer w-full px-4 py-2 border border-input-border bg-input-background text-input-foreground text-base ring-offset-0 transition-colors
		focus:border-secondary focus:outline-0 focus:ring focus:ring-secondary
		dark:focus:ring-foreground/10
		sm:text-2xs';
    $input_checkbox_base_class = 'lqd-input peer border-input-border
		focus:ring focus:ring-secondary
		dark:focus:ring-foreground/10';
    $input_checkbox_custom_wrapper_base_class = 'lqd-input-checkbox-custom-wrap inline-flex items-center justify-center size-[18px] shrink-0 rounded-full bg-foreground/10 text-heading-foreground bg-center bg-no-repeat
		peer-checked:bg-primary/[7%] peer-checked:text-primary';
    $label_base_class = 'lqd-input-label flex cursor-pointer items-center gap-2 text-2xs font-medium leading-none text-label';
    $label_extra_base_class = 'ms-auto';

    $variations = [
        'size' => [
            'none' => 'lqd-input-size-none',
            'sm' => 'lqd-input-sm h-9',
            'md' => 'lqd-input-md h-10',
            'lg' => 'lqd-input-lg h-11',
            'xl' => 'lqd-input-xl h-14 px-6',
        ]
    ];

    if ($type === 'textarea') {
        $size = 'none';
    }

    if ($switcher) {
        $input_checkbox_base_class .= ' lqd-input-switcher border-2 border-input-border [--input-rounded-multiplier:5] cursor-pointer appearance-none [background-size:1.3rem] bg-left bg-no-repeat transition-all
			checked:bg-right checked:bg-heading-foreground checked:border-heading-foreground
			dark:checked:bg-label dark:checked:border-label';

		if ( $switcherFill ) {
			$input_checkbox_base_class .= ' lqd-input-switcher-fill';
		}

        $variations['size'] = [
            'sm' => 'lqd-input-sm w-[34px] h-[18px]',
            'md' => 'lqd-input-md w-12 h-6',
        ];
    } elseif ($custom) {
        $input_checkbox_base_class = 'lqd-input peer rounded size-0 invisible absolute top-0 start-0';
    }

    if ($type !== 'checkbox' && $type !== 'radio') {
        $label_base_class .= ' mb-3';
    }

    if (!empty($label) && empty($id)) {
        $id = str()->random(10);
    }

    if ($stepper) {
        $input_base_class .= ' lqd-input-stepper appearance-none text-center px-2';
    }

    $size = isset($variations['size'][$size]) ? $variations['size'][$size] : $variations['size']['md'];
@endphp

<div
    {{ $attributes->withoutTwMergeClasses()->twMergeFor('container', $container_base_class, $containerClass, $attributes->get('class:container')) }}
    @if ($attributes->has('x-show')) x-show="{{ $attributes->get('x-show') }}" @endif
    @if ($type === 'password') x-data='{
		type: "{{ $type }}",
		get inputValueVisible() { return this.type !== "password" },
		toggleType() {
			this.type = this.type === "text" ? "password" : "text";
		}
    }' @endif
    @if ($stepper) x-data='{
		value: {{ !empty($value) ? $value : 0 }},
		min: {{ $attributes->has('min') ? $attributes->get('min') : 0 }},
		max: {{ $attributes->has('max') ? $attributes->get('max') : 999999 }},
		step: {{ $attributes->has('step') ? $attributes->get('step') : 1 }},
		setValue(value) {
			this.value = value;
			this.$refs.input.setAttribute("value", this.value);
			this.$refs.input.dispatchEvent(new Event("input"));
			this.$refs.input.dispatchEvent(new Event("change"));
		}
	}' @endif
>
    {{-- Label --}}
    @if (!empty($label) || ($type === 'checkbox' || $type === 'radio'))
        <label
            {{ $attributes->withoutTwMergeClasses()->twMergeFor('label', $label_base_class, $attributes->get('class:label')) }}
            for={{ $id }}
        >
            {{-- Checkbox and radio --}}
            @if ($type === 'checkbox' || $type === 'radio')
                <input
                    id="{{ $id }}"
                    {{ $attributes->withoutTwMergeClasses()->twMerge($input_checkbox_base_class, $size, $attributes->get('class')) }}
                    name="{{ $name }}"
                    type={{ $type }}
                    @if ($value) value={{ $value }} @endif
                    {{ $attributes }}
                    @if ($attributes->has('x-model')) x-model="{{ $attributes->get('x-model') }}" @endif
                >
                @if ($custom)
                    <span {{ $attributes->withoutTwMergeClasses()->twMergeFor('custom-wrap', $input_checkbox_custom_wrapper_base_class) }}></span>
                @endif
            @endif

            <span {{ $attributes->withoutTwMergeClasses()->twMergeFor('label-txt', 'lqd-input-label-txt', $attributes->get('class:label-txt')) }}>
                @if (!empty($labelIcon))
					{!! $labelIcon !!}
				@endif
				{{ $label }}
            </span>

            @if ($type === 'checkbox' || $type === 'radio')
                {{ $slot }}
            @endif

            @if (!empty($labelExtra))
                <span {{ $attributes->withoutTwMergeClasses()->twMergeFor('label-extra', $label_extra_base_class, $attributes->get('class:label-extra')) }}>
                    {{ $labelExtra }}
                </span>
            @endif

            {{-- Tooltip --}}
            @if (!empty($tooltip))
                <x-info-tooltip text="{{ $tooltip }}" />
            @endif
        </label>
    @endif

    {{-- Wrapper if there is icon over the input --}}
    @if ($type === 'password' || !empty($icon) || !empty($action) || $stepper)
        <div class="relative">
    @endif

    {{-- Inputs other than checkbox, radio and select --}}
    @if ($type !== 'checkbox' && $type !== 'radio' && $type !== 'select' && $type !== 'textarea' && $type !== 'color')
        <input
            id="{{ $id }}"
            {{ $attributes->withoutTwMergeClasses()->twMerge($input_base_class, $size, $attributes->get('class')) }}
            name="{{ $name }}"
            value="{{ $value }}"
            @if ($type === 'password') :type="type" @endif
            type={{ $type }}
            placeholder="{!! $placeholder !!}"
            {{ $attributes }}
            @if ($stepper) :value="(value).toString().includes('.') ? parseFloat(value).toFixed(2) : value" x-ref="input" @endif
            @if ($attributes->has('x-ref') && filled($attributes->get('x-ref'))) x-ref="{{ $attributes->get('x-ref') }}" @endif
            @if ($attributes->has('x-trap') && filled($attributes->get('x-trap'))) x-trap="{{ $attributes->get('x-trap') }}" @endif
            @if ($attributes->has('x-model')) x-model="{{ $attributes->get('x-model') }}" @endif
        />

        {{ $slot }}
    @endif

    {{-- Select input --}}
    @if ($type === 'select')
        <select
            id="{{ $id }}"
            {{ $attributes->withoutTwMergeClasses()->twMerge('cursor-pointer', $input_base_class, $size, $attributes->get('class')) }}
            name="{{ $name }}"
            value="{{ $value }}"
            placeholder="{!! $placeholder !!}"
            {{ $attributes }}
            @if ($attributes->has('x-model')) x-model="{{ $attributes->get('x-model') }}" @endif
        >
            {{ $slot }}
        </select>
    @endif

    {{-- Textarea input --}}
    @if ($type === 'textarea')
        <textarea
            id="{{ $id }}"
            {{ $attributes->withoutTwMergeClasses()->twMerge($input_base_class, $size, $attributes->get('class')) }}
            name="{{ $name }}"
            value="{{ $value }}"
            placeholder="{!! $placeholder !!}"
            {{ $attributes }}
            @if ($attributes->has('x-model')) x-model="{{ $attributes->get('x-model') }}" @endif
        >{{ $slot }}</textarea>
    @endif

    {{-- Color input --}}
    @if ($type === 'color')
        <div
            {{ $attributes->withoutTwMergeClasses()->twMerge($input_base_class, 'flex items-center gap-3', $size, $attributes->get('class')) }}
            x-data="liquidColorPicker({ colorVal: '{{ $value }}' })"
        >
            <div
                class="lqd-input-color-wrap relative size-5 shrink-0 gap-4 overflow-hidden rounded-full border shadow-sm focus-within:ring focus-within:ring-secondary"
                x-ref="colorInputWrap"
            >
                <span
                    class="absolute left-0 top-0 size-full"
                    :style="{ backgroundColor: colorVal }"
                ></span>
            </div>

			<input
				{{ $attributes->twMergeFor('input', 'grow border-none bg-transparent text-inherit outline-none') }}
				id="{{ $id }}"
				name="{{ $name }}"
				value="{{ $value }}"
				type="text"
				placeholder="{!! $placeholder !!}"
				:value="colorVal"
				x-ref="colorInput"
				@if ($attributes->has('x-model')) x-model="{{ $attributes->get('x-model') }}" @endif
				{{ $attributes }}
				@change="picker?.setColor($event.target.value);"
				@keydown.enter.prevent="picker?.setColor($event.target.value);"
				@focus="picker?.open(); $el.select()"
			/>
            <x-button
                {{ $attributes->twMergeFor('clear-btn', 'hidden text-2xs font-medium') }}
                variant="outline"
                size="sm"
                type="button"
                @click.prevent="picker?.clear()"
                ::class="{ 'hidden': colorVal === '' }"
            >
                @lang('Clear')
            </x-button>
        </div>
    @endif

    {{-- Password visibility toggle button --}}
    @if ($type === 'password')
        <button
            class="lqd-show-password absolute end-3 top-1/2 z-10 inline-flex size-7 -translate-y-1/2 cursor-pointer items-center justify-center rounded bg-none transition-colors hover:bg-foreground/10"
            type="button"
            @click="toggleType()"
        >
            <x-tabler-eye
                class="w-5"
                stroke-width="1.5"
                ::class="inputValueVisible ? 'hidden' : ''"
            />
            <x-tabler-eye-off
                class="hidden w-5"
                stroke-width="1.5"
                ::class="inputValueVisible ? '!block' : 'hidden'"
            />
        </button>
    @endif

    {{-- Icon --}}
    @if (!empty($icon))
        {!! $icon !!}
    @endif

    {{-- Action --}}
    @if (!empty($action))
        <div
			{{ $attributes->twMergeFor('action', "absolute inset-y-0 end-0 border-s") }}
		>
            {{ $action }}
        </div>
    @endif

    {{-- Stepper --}}
    @if ($stepper)
        <button
            class="lqd-stepper-btn absolute start-0 top-0 inline-flex aspect-square h-full w-10 items-center justify-center rounded-s-input transition-colors hover:bg-heading-foreground hover:text-heading-background"
            type="button"
            @click="setValue(Math.max(min, value - step))"
        >
            <x-tabler-minus
                class="w-4"
                stroke-width="1.5"
            />
        </button>
        <button
            class="lqd-stepper-btn absolute end-0 top-0 inline-flex aspect-square h-full w-10 items-center justify-center rounded-e-input transition-colors hover:bg-heading-foreground hover:text-heading-background"
            type="button"
            @click="setValue(Math.min(max, value + step))"
        >
            <x-tabler-plus
                class="w-4"
                stroke-width="1.5"
            />
        </button>
    @endif

    {{-- Wrapper if there is icon over the input --}}
    @if ($type === 'password' || !empty($icon) || !empty($action) || $stepper)
</div>
@endif
</div>

@if ($type === 'color')
    @pushOnce('script')
        <link
            rel="stylesheet"
            href="{{ custom_theme_url('assets/libs/jscolorpicker/dist/colorpicker.css') }}"
        >
        <script src="{{ custom_theme_url('assets/libs/jscolorpicker/dist/colorpicker.iife.min.js') }}"></script>
	@endPushOnce
@endif

{{-- Initiate Select2 for select elements with multiple attribute --}}
@if ($type === 'select' && $attributes->has('multiple'))
	@pushOnce('css')
	<link
		href="{{ custom_theme_url('/assets/libs/tom-select/dist/css/tom-select.min.css') }}"
		rel="stylesheet"
	/>
	@endPushOnce

    @pushOnce('script')
        <script src="{{ custom_theme_url('/assets/libs/tom-select/dist/js/tom-select.complete.min.js') }}"></script>

        <script>
            (() => {
                document.addEventListener('alpine:init', () => {
                    _.defer(() => {

                        const allSelectElements = document.querySelectorAll('select[multiple]');

                        allSelectElements.forEach(el => {
                            el.style.display = 'none';
                            const elId = el.id;
							const label = el.closest('.lqd-input-container')?.querySelector('.lqd-input-label');

                            const tomSelect = new TomSelect(el, {
                                create: {{ $addNew ? 'true' : 'false' }} && elId === '{{ $id }}',
                                plugins: {
                                    remove_button: {
                                        title: '{{ __('Remove this item') }}',
                                    }
                                },
                            });

							const buttonsWrapper = document.createElement('div');
							buttonsWrapper.classList.add('flex', 'flex-wrap', 'items-center', 'gap-2.5', 'mt-1');

                            const selectAllBtn = document.createElement('button');
							selectAllBtn.classList.add('text-[12px]', 'underline', 'font-medium');
                            selectAllBtn.setAttribute('type', 'button');
                            selectAllBtn.innerText = '{{ __('Select All') }}';

                            const deselectAllBtn = document.createElement('button');
							deselectAllBtn.classList.add('text-[12px]', 'underline', 'font-medium', 'hidden');
                            deselectAllBtn.setAttribute('type', 'button');
                            deselectAllBtn.innerText = '{{ __('Clear Selection') }}';

							selectAllBtn.addEventListener('click', () => {
								let values = Object.keys(tomSelect.options);
								tomSelect.addItems(values);
							});
							deselectAllBtn.addEventListener('click', () => {
								tomSelect.clear();
							});

							if ( !Object.keys(tomSelect.options).length ) {
								buttonsWrapper.classList.add('hidden');
							}

							if ( label ) {
								buttonsWrapper.classList.remove('mt-1');
								buttonsWrapper.classList.add('ms-auto');
								label.insertAdjacentElement('beforeend', buttonsWrapper);
							} else {
								tomSelect.wrapper.insertAdjacentElement('afterend', buttonsWrapper);
							}

                            buttonsWrapper.appendChild(selectAllBtn);
                            buttonsWrapper.appendChild(deselectAllBtn);

							tomSelect.on('change', () => {
								const total = Object.keys(tomSelect.options).length;
								const currentValues = tomSelect.getValue();

								buttonsWrapper.classList.toggle('hidden', !currentValues.length);
								selectAllBtn.classList.toggle('hidden', currentValues.length === total);
								deselectAllBtn.classList.toggle('hidden', !currentValues.length);
							})
                        })
                    })
                })
            })
            ();
        </script>
    @endPushOnce
@endif
