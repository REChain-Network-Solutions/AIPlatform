@extends('panel.layout.settings', ['disable_tblr' => true])
@section('title', isset($client) ? __('Edit Client') : __('Create New Client'))
@section('titlebar_actions', '')

@section('settings')
    <form
        class="flex flex-col gap-5"
        id="item_edit_form"
        onsubmit="return clientSave({{ $client->id ?? null }});"
        enctype="multipart/form-data"
    >
		@if (isset($client))
			@php
				$avatarSrc = custom_theme_url('assets/img/auth/default-avatar.png');

				if (isset($client) && $client->avatar) {
					if (str_starts_with($client->avatar, 'asset')) {
						$avatarSrc = custom_theme_url($client->avatar);
					} else {
						$avatarSrc = url('') . '/clientAvatar/' . $client->avatar;
					}
				}
			@endphp

			<img
				class="size-12 rounded-full object-cover object-center"
				src="{{ $avatarSrc }}"
				alt="Avatar"
			/>
		@endif

        <x-forms.input
            id="avatar"
            type="file"
            name="avatar"
            size="lg"
            label="{{ __('Avatar') }}"
            value="{{ isset($client) ? $client->avatar : null }}"
            accept="image/*"
        />

        <x-forms.input
            id="client_alt"
            name="client_alt"
            size="lg"
            label="{{ __('Alt') }}"
            value="{{ isset($client) ? $client->alt : null }}"
            required
        />

        <x-forms.input
            id="client_title"
            name="client_title"
            size="lg"
            label="{{ __('Title') }}"
            value="{{ isset($client) ? $client->title : null }}"
            required
        />

        <x-button
            id="item_edit_button"
            size="lg"
            type="submit"
        >
            {{ __('Save') }}
        </x-button>
    </form>
@endsection

@push('script')
    <script src="{{ custom_theme_url('/assets/js/panel/client.js') }}"></script>
@endpush
