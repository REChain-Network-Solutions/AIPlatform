<x-forms.input
	class:container="mb-2"
	id="ai_realtime_image"
	name="ai_realtime_image"
	type="checkbox"
	:checked="setting('ai_realtime_image', '1') == '1'"
	label="{{ __('AI Realtime Image') }}"
	switcher
/>
