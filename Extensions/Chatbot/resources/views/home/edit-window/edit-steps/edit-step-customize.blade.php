@php
    $predefined_colors = ['272733', '67D97C', 'E7AC47', '9D74C9', '017BE5'];
@endphp

{{-- Editing Step 2 - Customize --}}
<div
    class="col-start-1 col-end-1 row-start-1 row-end-1 transition-all"
    data-step="2"
    x-show="editingStep === 2"
    x-transition:enter-start="opacity-0 -translate-x-3"
    x-transition:enter-end="opacity-100 translate-x-0"
    x-transition:leave-start="opacity-100 translate-x-0"
    x-transition:leave-end="opacity-0 translate-x-3"
>
    <h2 class="mb-3.5">
        @lang('Customize')
    </h2>
    <p class="text-xs/5 opacity-60 lg:max-w-[360px]">
        @lang('Create and configure a chatbot that interacts with your users, ensuring it delivers accurate information.')
    </p>

    <div class="flex flex-col gap-7 pt-9">
        <div x-data="logoPreviewHandler">
            <x-forms.input
                class:label="text-heading-foreground"
                label="{{ __('Upload Logo') }}"
                name="logo"
                type="file"
                size="lg"
                x-bind="logoPicker"
                @change="externalChatbot && externalChatbot.toggleWindowState('open');"
            />
            <input
                type="hidden"
                x-model="uploadedLogo"
                x-modelable="activeChatbot.logo"
            >

            <template
                x-for="(error, index) in formErrors.logo"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

		<div>
			<x-forms.input
				class:label="text-heading-foreground"
				label="{{ __('Footer Link') }}"
				placeholder="{{ request()->getSchemeAndHttpHost() }}"
				name="footer_link"
				size="lg"
				x-model="activeChatbot.footer_link"
				@input.throttle.250ms="externalChatbot && externalChatbot.toggleWindowState('close')"
			/>

			<template
				x-for="(error, index) in formErrors.footer_link"
				:key="'error-' + index"
			>
				<div class="mt-2 text-2xs/5 font-medium text-red-500">
					<p x-text="error"></p>
				</div>
			</template>
		</div>

        <div>
            <label
                class="mb-5 block w-full cursor-pointer text-xs font-medium text-heading-foreground"
                for="avatar-{{ $avatars[0]['id'] }}"
            >
                @lang('Avatar')
            </label>
            <p class="mb-5 w-full text-sm">
                @lang('Select an avatar for your chatbot.')
            </p>
            <div class="grid grid-cols-3 gap-x-2 gap-y-3 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
                <div class="lqd-chatbot-avatar-list contents">
                    @foreach ($avatars as $avatar)
                        <div class="lqd-chatbot-avatar-list-item relative flex justify-center">
                            <input
                                class="size-0 peer invisible"
                                id="avatar-{{ $avatar['id'] }}"
                                type="radio"
                                name="avatar"
                                value="{{ $avatar['avatar'] }}"
                                x-model="activeChatbot.avatar"
                            />
                            <label
                                class="size-10 relative inline-grid cursor-pointer place-items-center transition-all hover:scale-110 peer-checked:drop-shadow-xl"
                                for="avatar-{{ $avatar['id'] }}"
                                tabindex="0"
                            >
                                <img
                                    width="40"
                                    height="40"
                                    src="{{ $avatar['avatar_url'] }}"
                                >
                            </label>
                            <span
                                class="size-6 pointer-events-none invisible absolute start-1/2 top-1/2 inline-grid -translate-x-1/2 -translate-y-1/2 scale-75 place-items-center rounded-full bg-white/15 text-heading-foreground opacity-0 backdrop-blur-md backdrop-saturate-150 transition-all peer-checked:visible peer-checked:scale-100 peer-checked:opacity-100"
                            >
                                <x-tabler-check class="size-[18px] text-white" />
                            </span>
                        </div>
                    @endforeach
                </div>
                <div
                    class="relative flex justify-center"
                    x-data="customAvatar"
                >
                    <input
                        class="size-0 peer invisible"
                        id="avatar-custom"
                        type="file"
                        x-bind="customAvatarPicker"
                    >
                    <label
                        class="size-10 inline-grid cursor-pointer place-items-center rounded-full bg-heading-foreground/5 text-heading-foreground transition-all hover:scale-110 hover:bg-heading-foreground hover:text-heading-background peer-checked:drop-shadow-xl"
                        for="avatar-custom"
                        tabindex="0"
                    >
                        <x-tabler-plus class="size-4" />
                    </label>
                </div>
            </div>

            <template
                x-for="(error, index) in formErrors.avatar"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        {{-- <x-forms.input
            class:label="text-heading-foreground"
            class="capitalize"
            size="lg"
            type="select"
            name="color_mode"
            label="{{ __('Color Mode') }}"
        >
            @foreach (\App\Extensions\Chatbot\Src\Enums\ColorModeEnum::toArray() as $mode)
                <option value="{{ $mode }}">{{ $mode }}</option>
            @endforeach
        </x-forms.input> --}}

        <div>
            <label
                class="mb-5 block w-full cursor-pointer text-xs font-medium text-heading-foreground"
                for="color-{{ $predefined_colors[0] }}"
            >
                @lang('Color')
            </label>
            <p class="mb-5 w-full text-sm">
                @lang('Choose an accent color that represents your brand.')
            </p>
            <div class="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
                @foreach ($predefined_colors as $color)
                    <div class="relative flex justify-center">
                        <input
                            class="size-0 peer invisible"
                            id="color-{{ $color }}"
                            type="radio"
                            name="color"
                            value="#{{ $color }}"
                            x-model="activeChatbot.color"
                        />
                        <label
                            class="size-10 relative inline-grid cursor-pointer place-items-center rounded-full border-[3px] border-background shadow-[0_4px_12px_rgba(0,0,0,0.11)] transition-all hover:scale-110 peer-checked:shadow-xl"
                            style="background-color: #{{ $color }}"
                            for="color-{{ $color }}"
                            tabindex="0"
                        >
                        </label>
                        <x-tabler-check
                            class="size-[18px] pointer-events-none absolute start-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 scale-50 text-white opacity-0 transition-all peer-checked:scale-100 peer-checked:opacity-100"
                        />
                    </div>
                @endforeach
                <div
                    class="relative flex justify-center"
                    x-data="customColorPicker"
                >
                    <input
                        class="size-0 peer invisible"
                        id="color-custom"
                        name="color"
                        type="radio"
                        value=""
                        x-bind="customColorRadioInput"
                    >
                    <input
                        class="size-0 invisible absolute"
                        id="custom-color-input"
                        type="color"
                        x-bind="customColorColorInput"
                        x-model="activeChatbot.color"
                    />
                    <label
                        class="size-10 relative inline-grid cursor-pointer place-items-center rounded-full border-[3px] border-background text-heading-foreground shadow-[0_4px_12px_rgba(0,0,0,0.11)] transition-all after:absolute after:inset-0 after:rounded-full after:bg-black/10 hover:scale-110 hover:bg-heading-foreground hover:text-heading-background peer-checked:drop-shadow-xl"
                        for="color-custom"
                        tabindex="0"
                        style="background: conic-gradient(from 90deg, violet, indigo, blue, green, yellow, orange, red, violet);"
                        x-bind="customColorTrigger"
                    >
                    </label>
                    <x-tabler-check
                        class="size-[18px] pointer-events-none absolute start-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 scale-50 text-white opacity-0 transition-all peer-checked:scale-100 peer-checked:opacity-100"
                    />
                    <span
                        class="size-6 absolute end-0 top-0 inline-block -translate-y-1/3 translate-x-1/3 scale-50 rounded-full border border-background opacity-0 shadow-lg transition-all peer-checked:scale-100 peer-checked:opacity-100"
                        x-bind="customColorOutput"
                    ></span>
                </div>
            </div>

            <template
                x-for="(error, index) in formErrors.color"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        <div>
            <x-forms.input
                class="h-[18px] w-[34px] [background-size:0.625rem]"
                class:label="flex-row-reverse justify-between text-xs font-medium text-heading-foreground"
                containerClass="[&_.lqd-input-label-txt]:order-1 [&_.lqd-tooltip-container]:me-auto"
                label="{{ __('Show Logo') }}"
                type="checkbox"
                switcher
                name="show_logo"
                ::checked="activeChatbot.show_logo === 1"
                x-model.boolean="activeChatbot.show_logo"
                @change="externalChatbot.toggleWindowState('open')"
            />
            <template
                x-for="(error, index) in formErrors.show_logo"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        <div>
            <x-forms.input
                class="h-[18px] w-[34px] [background-size:0.625rem]"
                class:label="flex-row-reverse justify-between text-xs font-medium text-heading-foreground"
                containerClass="[&_.lqd-input-label-txt]:order-1 [&_.lqd-tooltip-container]:me-auto"
                label="{{ __('Show Date and Time') }}"
                type="checkbox"
                switcher
                name="show_date_and_time"
                ::checked="activeChatbot.show_date_and_time === 1"
                x-model.boolean="activeChatbot.show_date_and_time"
                @change="if ( externalChatbot ) { externalChatbot.toggleWindowState('open'); externalChatbot.toggleView('conversation-messages') }"
            />

            <template
                x-for="(error, index) in formErrors.show_date_and_time"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        {{-- <x-forms.input
            class="h-[18px] w-[34px] [background-size:0.625rem]"
            class:label="flex-row-reverse justify-between text-xs font-medium text-heading-foreground"
            containerClass="[&_.lqd-input-label-txt]:order-1 [&_.lqd-tooltip-container]:me-auto"
            label="{{ __('Show Avarage Response Time') }}"
            type="checkbox"
            switcher
            name="show_average_response_time"
            ::checked="activeChatbot.show_average_response_time === 1"
            x-model.boolean="activeChatbot.show_average_response_time"
            @change="if ( externalChatbot ) { externalChatbot.toggleWindowState('open'); externalChatbot.toggleView('conversation-messages') }"
        /> --}}

        <div>
            <x-forms.input
                class="h-[18px] w-[34px] [background-size:0.625rem]"
                class:label="flex-row-reverse justify-between text-xs font-medium text-heading-foreground"
                containerClass="[&_.lqd-input-label-txt]:order-1 [&_.lqd-tooltip-container]:me-auto"
                label="{{ __('Transparent Trigger') }}"
                type="checkbox"
                switcher
                name="trigger_background"
                tooltip="{{ __('If activated, the trigger will be transparent in idle state. But we add a background to the trigger in active mode to make it more visible.') }}"
                ::checked="activeChatbot.trigger_background === 'transparent'"
                @change="
                    $event.target.checked ? activeChatbot.trigger_background = 'transparent' : activeChatbot.trigger_background = '';
                    externalChatbot && externalChatbot.toggleWindowState('close');
                "
                x-model="activeChatbot.trigger_background"
            />

            <template
                x-for="(error, index) in formErrors.trigger_background"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        <div
            class="flex w-full flex-col gap-5 text-heading-foreground"
            x-data="{ currentVal: `${parseInt(activeChatbot.trigger_avatar_size || 60, 10)}px` }"
        >
            <label
                class="block w-full text-xs font-medium text-heading-foreground"
                for="trigger_avatar_size"
            >
                @lang('Trigger Avatar Size')
            </label>
            <div class="flex items-center gap-3">
                <input
                    class="[&::-moz-range-thumb]:size-4 [&::-webkit-slider-thumb]:size-4 h-2 w-full cursor-ew-resize appearance-none rounded-full bg-heading-foreground/5 focus:outline-secondary [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-heading-foreground active:[&::-moz-range-thumb]:scale-110 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-heading-foreground active:[&::-webkit-slider-thumb]:scale-110"
                    type="range"
                    min="20"
                    max="100"
                    step="1"
                    name="trigger_avatar_size"
                    @input="
                        currentVal = `${$event.target.value}px`;
                        activeChatbot.trigger_avatar_size = currentVal;
                        externalChatbot && externalChatbot.toggleWindowState('close');
                    "
                    x-modelable="currentVal"
                />
                <span
                    class="min-w-10 ms-2 shrink-0 text-2xs font-medium"
                    x-text="parseInt(currentVal, 10) + 'px'"
                ></span>
            </div>

            <template
                x-for="(error, index) in formErrors.trigger_avatar_size"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>

        <div>
            <label
                class="mb-5 block w-full cursor-pointer text-xs font-medium text-heading-foreground"
                for="position-left"
            >
                @lang('Position')
            </label>
            <div class="flex flex-wrap justify-between gap-2">
                @foreach (\App\Extensions\Chatbot\System\Enums\PositionEnum::toArray() as $position)
                    <div class="relative text-center">
                        <input
                            class="size-0 peer invisible absolute"
                            id="position-{{ $position }}"
                            type="radio"
                            name="position"
                            value="{{ $position }}"
                            x-model="activeChatbot.position"
                        />
                        <label
                            @class([
                                'h-[105px] w-[150px] rounded-lg bg-heading-foreground/5 p-3 text-heading-foreground/20 flex items-end cursor-pointer transition-all hover:scale-105',
                                'justify-end' => $position === 'right',
                            ])
                            for="position-{{ $position }}"
                        >
                            <svg
                                @class([
                                    '-scale-x-100' => $position === 'right',
                                ])
                                width="37"
                                height="50"
                                viewBox="0 0 37 50"
                                fill="currentColor"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M0 45.0721C0 42.863 1.79086 41.0721 4 41.0721C6.20914 41.0721 8 42.863 8 45.0721C8 47.2813 6.20914 49.0721 4 49.0721C1.79086 49.0721 0 47.2813 0 45.0721Z"
                                />
                                <path
                                    d="M0 4.07214C0 1.863 1.79086 0.0721436 4 0.0721436H33C35.2091 0.0721436 37 1.863 37 4.07214V33.0721C37 35.2813 35.2091 37.0721 33 37.0721H4C1.79086 37.0721 0 35.2813 0 33.0721V4.07214Z"
                                />
                            </svg>
                        </label>
                        <span
                            class="size-7 pointer-events-none absolute end-1.5 top-1.5 inline-grid scale-110 place-items-center rounded-full bg-primary/10 text-primary opacity-0 transition-all peer-checked:scale-100 peer-checked:opacity-100"
                        >
                            <x-tabler-check class="size-5" />
                        </span>
                        <span class="mt-5 block text-xs font-semibold capitalize text-heading-foreground">
                            {{ $position }}
                        </span>
                    </div>
                    @if (!$loop->last)
                        <span class="w-px bg-heading-foreground/10"></span>
                    @endif
                @endforeach
            </div>

            <template
                x-for="(error, index) in formErrors.position"
                :key="'error-' + index"
            >
                <div class="mt-2 text-2xs/5 font-medium text-red-500">
                    <p x-text="error"></p>
                </div>
            </template>
        </div>
    </div>
</div>
