@extends('panel.layout.app', ['disable_tblr' => true])
@section('title', __('BrandCenter'))
@section('titlebar_actions', '')

@section('content')
    <div class="py-10">

        <div class="flex flex-wrap justify-between gap-y-8">
            <x-card class="w-full lg:w-[48%]">
                <h2 class="mb-7 flex flex-wrap items-center justify-between gap-3">
                    {{ __('Twitter/X') }}
                    @if ($platformX)
                        <x-button
                            class="ms-auto"
                            variant="secondary"
                            href="{{ route('dashboard.user.automation.platform.disconnect', $platformX->id) }}"
                        >
                            {{ __('Disconnect') }}
                        </x-button>
                    @endif
                </h2>

                <form
                    class="flex flex-col gap-6"
					method="post"
					action="{{ route('dashboard.user.automation.platform.update', \App\Extensions\AISocialMedia\System\Enums\Platform::x->value) }}"
                >
					@csrf
                    <a
                        class="underline"
                        href="https://developer.x.com/en/docs/apps/overview"
                        target="_blank"
                    >
                        {{ __('X Developer Portal') }}</a>
                    <x-forms.input
                        id="consumer_key"
                        label="{{ __('API Key') }}"
                        name="consumer_key"
                        size="lg"
                        required
                        value="{{ $platformX?->getCredentialValue('consumer_key')}}"
                    />

                    <x-forms.input
                        id="consumer_secret"
                        label="{{ __('API Key Secret') }}"
                        name="consumer_secret"
                        size="lg"
                        required
                        value="{{  $platformX?->getCredentialValue('consumer_secret') }}"
                    />

                    <x-forms.input
                        id="bearer_token"
                        label="{{ __('Bearer Token') }}"
                        name="bearer_token"
                        size="lg"
                        required
						value="{{  $platformX?->getCredentialValue('bearer_token') }}"
                    />

                    <x-forms.input
                        id="access_token"
                        label="{{ __('Access Token') }}"
                        name="access_token"
                        size="lg"
                        required
						value="{{  $platformX?->getCredentialValue('access_token') }}"
                    />

                    <x-forms.input
                        id="access_token_secret"
                        label="{{ __('Access Token Secret') }}"
                        name="access_token_secret"
                        size="lg"
                        required
						value="{{  $platformX?->getCredentialValue('access_token_secret') }}"
                    />

                    <x-forms.input
                        id="account_id"
                        label="{{ __('Client ID') }}"
                        name="account_id"
                        size="lg"
                        required
						value="{{  $platformX?->getCredentialValue('account_id') }}"
                    />


					@if($platformX)
						<hr>
						<x-forms.input
							disabled
							label="{{ __('Expired At') }}"
							name="expires_at"
							size="lg"
							required
							value="{{  $platformX?->getAttribute('expires_at') }}"
						/>
					@endif

                    <x-button
                        id="twitter_button"
                        type="submit"
                        size="lg"
                    >
                        {{ __('Update') }}
                    </x-button>
                </form>
            </x-card>

            <x-card class="w-full lg:w-[48%]">
                <h2 class="mb-7 flex flex-wrap items-center justify-between gap-3">
                    {{ __('LinkedIn') }}
                    @if ($platformLinkedin)
                        <x-button
                            class="ms-auto"
                            variant="secondary"
                            href="{{ route('dashboard.user.automation.platform.disconnect', $platformLinkedin->id) }}"
                        >
                            {{ __('Disconnect') }}
                        </x-button>
                    @endif
                </h2>

				<form
					class="flex flex-col gap-6"
					method="post"
					action="{{ route('dashboard.user.automation.platform.update', \App\Extensions\AISocialMedia\System\Enums\Platform::linkedin->value) }}"
				>
					@csrf
                    <a
                        class="underline"
                        href="https://www.linkedin.com/developers/tools/oauth"
                        target="_blank"
                    >
                        {{ __('OAuth token generator tool') }}</a>
                    <x-forms.input
                        id="access_token"
                        label="{{ __('Your Access Token') }}"
                        name="access_token"
                        size="lg"
                        required
                        value="{{ $platformLinkedin?->getCredentialValue('access_token') }}"
                    />

					@if($platformLinkedin)
						<hr>
						<x-forms.input
							disabled
							label="{{ __('Expired At') }}"
							name="expires_at"
							size="lg"
							required
							value="{{  $platformLinkedin?->getAttribute('expires_at') }}"
						/>
					@endif

                    <x-button
                        id="linkedin_button"
                        size="lg"
                        type="submit"
                    >
                        {{ __('Update') }}
                    </x-button>
                </form>
            </x-card>

            <x-card class="w-full lg:w-[48%]">
                <h2 class="mb-7 flex flex-wrap items-center justify-between gap-3">
                    {{ __('Instagram') }}
                    @if ($platformInstagram)
                        <x-button
                            class="ms-auto"
                            variant="secondary"
                            href="{{ route('dashboard.user.automation.platform.disconnect', $platformInstagram->id) }}"
                        >
                            {{ __('Disconnect') }}
                        </x-button>
					@else
						<x-button
							target="_blank"
							class="ms-auto"
							variant="success"
							href="{{ url('oauth/redirect/instagram') }}"
						>
							{{ __('Connect to instagram') }}
						</x-button>
					@endif
                </h2>

				<form
					class="flex flex-col gap-6"
				>
					@csrf
                    <x-forms.input
						disabled
                        label="{{ __('Name') }}"
                        size="lg"
                        required
                        value="{{ $platformInstagram?->getCredentialValue('name') }}"
                    />
                    <x-forms.input
						disabled
                        label="{{ __('Username') }}"
                        size="lg"
                        required
                        value="{{ $platformInstagram?->getCredentialValue('username') }}"
                    />


					@if($platformInstagram)
						<hr>
						<x-forms.input
							disabled
							label="{{ __('Expired At') }}"
							name="expires_at"
							size="lg"
							required
							value="{{  $platformInstagram?->getAttribute('expires_at') }}"
						/>
					@endif
					@if(\Illuminate\Support\Facades\Auth::user()->isAdmin())
						<x-alert class="mt-2 mb-3">
							<p>
								{!! trans('To run Instagram, first adjust the settings. You can adjust the settings from <a class="text-red-600" href="/dashboard/admin/automation/settings">here</a>.') !!}
							</p>
						</x-alert>
					@endif
                </form>
            </x-card>
        </div>
    </div>
@endsection
