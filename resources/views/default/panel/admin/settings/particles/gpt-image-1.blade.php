<div class="col-md-12">
	<div class="mb-3">
		<x-card
			class="w-full"
			size="sm"
		>
			<x-forms.input
				id="enabled_gpt_image_1"
				type="checkbox"
				switcher
				type="checkbox"
				:checked="setting('enabled_gpt_image_1', 0) == 1"
				label="{{ __('Enabled GPT-IMAGE-1 model') }}"
			>
				<x-badge
					class="ms-2 text-2xs"
					variant="secondary"
				>
					@lang('New')
				</x-badge>
			</x-forms.input>
		</x-card>
	</div>
</div>
