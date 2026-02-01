@php use App\Domains\Entity\Enums\EntityEnum; @endphp
@extends('panel.layout.settings', ['layout' => 'wide'])
@section('title', __('PiAPI Settings'))
@section('titlebar_actions', '')
@section('titlebar_subtitle', __('This API key is used for these features: AI Image'))

@section('additional_css')
    <link
        href="{{ custom_theme_url('/assets/libs/select2/select2.min.css') }}"
        rel="stylesheet"
    />
@endsection

@section('settings')
    <form
        method="post"
        action="{{ route('dashboard.admin.settings.piapi-ai') }}"
        id="settings_form"
        enctype="multipart/form-data"
    >
        @csrf
        <h3 class="mb-[25px] text-[20px]">{{ __('PiAPI Settings') }}</h3>
        <div class="row">
            <!-- TODO OPENAI API KEY -->
            <x-card
                class="mb-3 max-md:text-center"
                szie="lg"
            >

                <div class="col-md-12">
                    <div
                        class="form-control mb-3 border-none p-0 [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius] [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em]">
                        <label class="form-label">{{ __('PiAPI API key') }}</label>

                        <select
                            class="form-control select2"
                            id="piapi_ai_api_secret"
                            name="piapi_ai_api_secret"
                            multiple
                        >
                            @if($app_is_demo)
                                <option selected value="*********************">*********************</option>
                            @else
                                @foreach (explode(',', setting('piapi_ai_api_secret','api key')) as $secret)
                                    <option
                                        value="{{ $secret }}"
                                        selected
                                    >{{ $secret }}</option>
                                @endforeach
                            @endif
                        </select>
                        <x-alert class="mt-2">
                            <p>
                                {{ __('You can enter as much API key as you want. Click "Enter" after each API key.') }}
                            </p>
                        </x-alert>
                        <x-alert class="mt-2">
                            <p>
                                {{ __('Please ensure that your PiAPI API key is fully functional and billing defined on your PiAPI account.') }}
                            </p>
                        </x-alert>
                    </div>
                </div>

				<div class="col-md-12">
					<div class="col-md-12">
						<div class="mb-3">
							<x-card
								class="w-full"
								size="sm"
							>
								<x-forms.input
									tooltip="{{ trans('If you enable this setting, multiple variations matching your prompt will also be generated.') }}"
									id="midjourney_variation"
									name="midjourney_variation"
									type="checkbox"
									switcher
									type="checkbox"
									:checked="setting('midjourney_variation', '0') == '1'"
									label="{{ __('Generate Multiple Variations') }}"
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
				</div>
            </x-card>
        </div>
        <button
            class="btn btn-primary w-full"
            type="submit"
        >
            {{ __('Save') }}
        </button>
    </form>
@endsection
@push('script')
    <script src="{{ custom_theme_url('/assets/js/panel/settings.js') }}"></script>
    <script src="{{ custom_theme_url('/assets/libs/select2/select2.min.js') }}"></script>
@endpush
