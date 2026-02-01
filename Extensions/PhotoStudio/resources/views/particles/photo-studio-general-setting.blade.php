<x-forms.input
        class:container="mb-2"
        id="photo_studio"
		name="photo_studio"
        type="checkbox"
        :checked="setting('photo_studio') == 1 || setting('photo_studio') == null"
        label="{{ __('Photo Studio') }}"
        switcher
/>
