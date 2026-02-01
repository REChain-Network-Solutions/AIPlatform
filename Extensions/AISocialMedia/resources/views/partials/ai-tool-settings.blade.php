<x-forms.input
	class:container="mb-2"
	id="ai_automation"
	name="ai_automation"
	type="checkbox"
	:checked="setting('ai_automation', '1') == '1'"
	label="{{ __('AI Automation') }}"
	switcher
/>
