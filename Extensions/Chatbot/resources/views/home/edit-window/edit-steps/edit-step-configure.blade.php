{{-- Editing Step 1 - Configure --}}
<div
	class="col-start-1 col-end-1 row-start-1 row-end-1 transition-all"
	data-step="1"
	x-show="editingStep === 1"
	x-transition:enter-start="opacity-0 -translate-x-3"
	x-transition:enter-end="opacity-100 translate-x-0"
	x-transition:leave-start="opacity-100 translate-x-0"
	x-transition:leave-end="opacity-0 translate-x-3"
>
	<h2 class="mb-3.5">
		@lang('Configure')
	</h2>
	<p class="text-xs/5 opacity-60 lg:max-w-[360px]">
		@lang('Create and configure a chatbot that interacts with your users, ensuring it delivers accurate information.')
	</p>

	<div class="flex flex-col gap-7 pt-9">
		<div>
			<x-forms.input
				class:label="text-heading-foreground"
				label="{{ __('Chatbot Title') }}"
				placeholder="{{ __('MagicBot') }}"
				name="title"
				size="lg"
				x-model="activeChatbot.title"
				@input.throttle.250ms="externalChatbot && externalChatbot.toggleWindowState('open')"
			/>

			<template
				x-for="(error, index) in formErrors.title"
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
				label="{{ __('Bubble Message') }}"
				placeholder="{{ __('MagicBot') }}"
				name="bubble_message"
				size="lg"
				x-model="activeChatbot.bubble_message"
				@input.throttle.250ms="externalChatbot && externalChatbot.toggleWindowState('close')"
			/>

			<template
				x-for="(error, index) in formErrors.bubble_message"
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
				label="{{ __('Welcome Message') }}"
				placeholder="{{ __('Enter welcome message') }}"
				name="welcome_message"
				size="lg"
				x-model="activeChatbot.welcome_message"
				@input.throttle.250ms="if ( externalChatbot ) { externalChatbot.toggleWindowState('open'); externalChatbot.toggleView('conversation-messages') }"
			/>

			<template
				x-for="(error, index) in formErrors.welcome_message"
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
				label="{{ __('Chatbot Instructions') }}"
				placeholder="{{ __('Explain chatbot role') }}"
				name="instructions"
				size="lg"
				type="textarea"
				rows="5"
				x-model="activeChatbot.instructions"
			/>

			<template
				x-for="(error, index) in formErrors.instructions"
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
				class:label="text-heading-foreground flex-row-reverse justify-between"
				label="{{ __('Do Not Go Beyond Instructions') }}"
				name="do_not_go_beyond_instructions"
				size="lg"
				type="checkbox"
				switcher
				x-model.boolean="activeChatbot.do_not_go_beyond_instructions"
			/>

			<template
				x-for="(error, index) in formErrors.do_not_go_beyond_instructions"
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
				label="{{ __('Language') }}"
				name="language"
				size="lg"
				type="select"
				x-model="activeChatbot.language"
			>
				<option
					value="auto"
					selected
				>
					@lang('Auto')
				</option>
				@foreach (LaravelLocalization::getSupportedLocales() as $localeCode => $properties)
					@if (in_array($localeCode, explode(',', $settings_two->languages), true))
						<option value="{{ $localeCode }}">
							{{ $properties['name'] }}
						</option>
					@endif
				@endforeach
			</x-forms.input>
			<template
				x-for="(error, index) in formErrors.language"
				:key="'error-' + index"
			>
				<div class="mt-2 text-2xs/5 font-medium text-red-500">
					<p x-text="error"></p>
				</div>
			</template>
		</div>

		@includeIf('chatbot-agent::particles.chatbot-config')

		{{--        <div>--}}
		{{--            <x-forms.input--}}
		{{--                class:label="text-heading-foreground"--}}
		{{--                label="{{ __('AI Model') }}"--}}
		{{--                name="ai_model"--}}
		{{--                size="lg"--}}
		{{--                type="select"--}}
		{{--                x-model="activeChatbot.ai_model"--}}
		{{--                x-ref="aiModelSelect"--}}
		{{--            >--}}
		{{--                @foreach (\App\Domains\Entity\Enums\EntityEnum::reWriterModels(\App\Domains\Engine\Enums\EngineEnum::OPEN_AI) as $model)--}}
		{{--                    <option value="{{ $model->value }}">--}}
		{{--                        {{ $model->label() }}--}}
		{{--                    </option>--}}
		{{--                @endforeach--}}
		{{--            </x-forms.input>--}}

		{{--            <template--}}
		{{--                x-for="(error, index) in formErrors.ai_model"--}}
		{{--                :key="'error-' + index"--}}
		{{--            >--}}
		{{--                <div class="mt-2 text-2xs/5 font-medium text-red-500">--}}
		{{--                    <p x-text="error"></p>--}}
		{{--                </div>--}}
		{{--            </template>--}}
		{{--        </div>--}}
	</div>
</div>
