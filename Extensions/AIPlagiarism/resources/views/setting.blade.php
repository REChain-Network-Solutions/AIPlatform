@extends('panel.layout.app')
@section('title', __('Plagiarism API Settings'))
@section('titlebar_subtitle', __('This API key is used for these features: AI Plagiarism, AI Detector'))

@section('additional_css')
    <link href="/assets/select2/select2.min.css" rel="stylesheet" />
@endsection

@section('content')
    <!-- Page body -->
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-5 mx-auto">
                    <form id="settings_form" onsubmit="return plagiarismSettingsSave();" enctype="multipart/form-data">
                        <h3 class="mb-[25px] text-[20px]">{{ __('Plagiarism API Settings') }}</h3>
                        <div class="row">
                            {{-- <!-- TODO Plagiarism api key --> --}}
                            @if ($app_is_demo)
                                <div class="col-md-12">
                                    <div class="mb-3">
                                        <label class="form-label">{{ __('Plagiarism API Key') }}</label>
                                        <input type="text" class="form-control" id="plagiarism_api_key"
                                            name="plagiarism_api_key" value="*********************">
                                    </div>
                                </div>
                            @else
                                <div class="col-md-12">
                                    <div
                                        class="form-control border-none p-0 mb-3 [&_.select2-selection--multiple]:!border-[--tblr-border-color] [&_.select2-selection--multiple]:!p-[1em_1.23em] [&_.select2-selection--multiple]:!rounded-[--tblr-border-radius]">
                                        <label class="form-label">{{ __('Plagiarism API Key') }}</label>
                                        <input type="text" class="form-control" id="plagiarism_api_key"
                                            name="plagiarism_api_key" value="{{ $settings_two->plagiarism_key }}"
                                            required>
                                        {{-- <div
                                            class="bg-blue-100 text-blue-600 rounded-xl !p-3 !mt-2 dark:bg-blue-600/20 dark:text-blue-200">
                                            <svg class="inline !me-1" xmlns="http://www.w3.org/2000/svg" width="22"
                                                height="22" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
                                                fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                <path d="M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0"></path>
                                                <path d="M12 9h.01"></path>
                                                <path d="M11 12h1v4h1"></path>
                                            </svg> --}}
                                        {{-- {{ __('You can enter as much api key as you want. Click "Enter" after each api key.') }} --}}
                                        {{-- </div>  --}}
                                        <div
                                            class="bg-orange-100 text-orange-600 rounded-xl !p-3 !mt-2 dark:bg-orange-600/20 dark:text-orange-200">
                                            <svg class="inline !me-1" xmlns="http://www.w3.org/2000/svg" width="20"
                                                height="20" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor"
                                                fill="none" stroke-linecap="round" stroke-linejoin="round">
                                                <path
                                                    d="M10.24 3.957l-8.422 14.06a1.989 1.989 0 0 0 1.7 2.983h16.845a1.989 1.989 0 0 0 1.7 -2.983l-8.423 -14.06a1.989 1.989 0 0 0 -3.4 0z">
                                                </path>
                                                <path d="M12 9v4"></path>
                                                <path d="M12 17h.01"></path>
                                            </svg>
                                            {{ __('Please ensure that your Plagiarism api key is fully functional and billing defined on your Plagiarism account.') }}
                                        </div>
                                        {{-- <a href="https://serper.dev" target="_blank">{{ __('Get Perper Api Key') }}</a> --}}

                                        {{-- <a href="{{ route('dashboard.admin.settings.serperapi.test') }}" target="_blank"
                                            class="btn btn-primary w-100 mt-2 mb-2">
                                            {{ __('After Saving Setting, Click Here to Test Your api key') }}
                                        </a> --}}
                                    </div>
                                </div>
                            @endif

                            {{-- <div class="col-md-12">
								<div class="mb-3">
									<label class="form-label">{{__('Default stablediffusion Language')}}</label>
									<select class="form-select" name="stablediffusion_default_language" id="stablediffusion_default_language">
										<option value="ar-AE" {{$settings_two->stablediffusion_default_language == 'ar-AE' ? 'selected' : null}}>{{__('Arabic')}}</option>
										<option value="cmn-CN"{{$settings_two->stablediffusion_default_language == 'cmn-CN' ? 'selected' : null}}>{{__('Chinese (Mandarin)')}}</option>
										<option value="hr-HR" {{$settings_two->stablediffusion_default_language == 'hr-HR' ? 'selected' : null}}>{{__('Croatian (Croatia)')}}</option>
										<option value="cs-CZ" {{$settings_two->stablediffusion_default_language == 'cs-CZ' ? 'selected' : null}}>{{__('Czech (Czech Republic)')}}</option>
										<option value="da-DK" {{$settings_two->stablediffusion_default_language == 'da-DK' ? 'selected' : null}}>{{__('Danish (Denmark)')}}</option>
										<option value="nl-NL" {{$settings_two->stablediffusion_default_language == 'nl-NL' ? 'selected' : null}}>{{__('Dutch (Netherlands)')}}</option>
										<option value="en-US" {{$settings_two->stablediffusion_default_language == 'en-US' ? 'selected' : null}}>{{__('English (USA)')}}</option>
										<option value="et-EE" {{$settings_two->stablediffusion_default_language == 'et-EE' ? 'selected' : null}}>{{__('Estonian (Estonia)')}}</option>
										<option value="fi-FI" {{$settings_two->stablediffusion_default_language == 'fi-FI' ? 'selected' : null}}>{{__('Finnish (Finland)')}}</option>
										<option value="fr-FR" {{$settings_two->stablediffusion_default_language == 'fr-FR' ? 'selected' : null}}>{{__('French (France)')}}</option>
										<option value="de-DE" {{$settings_two->stablediffusion_default_language == 'de-DE' ? 'selected' : null}}>{{__('German (Germany)')}}</option>
										<option value="el-GR" {{$settings_two->stablediffusion_default_language == 'el-GR' ? 'selected' : null}}>{{__('Greek (Greece)')}}</option>
										<option value="he-IL" {{$settings_two->stablediffusion_default_language == 'he-IL' ? 'selected' : null}}>{{__('Hebrew (Israel)')}}</option>
										<option value="hi-IN" {{$settings_two->stablediffusion_default_language == 'hi-IN' ? 'selected' : null}}>{{__('Hindi (India)')}}</option>
										<option value="hu-HU" {{$settings_two->stablediffusion_default_language == 'hu-HU' ? 'selected' : null}}>{{__('Hungarian (Hungary)')}}</option>
										<option value="is-IS" {{$settings_two->stablediffusion_default_language == 'is-IS' ? 'selected' : null}}>{{__('Icelandic (Iceland)')}}</option>
										<option value="id-ID" {{$settings_two->stablediffusion_default_language == 'id-ID' ? 'selected' : null}}>{{__('Indonesian (Indonesia)')}}</option>
										<option value="it-IT" {{$settings_two->stablediffusion_default_language == 'it-IT' ? 'selected' : null}}>{{__('Italian (Italy)')}}</option>
										<option value="ja-JP" {{$settings_two->stablediffusion_default_language == 'ja-JP' ? 'selected' : null}}>{{__('Japanese (Japan)')}}</option>
										<option value="kk-KZ" {{$settings_two->stablediffusion_default_language == 'kk-KZ' ? 'selected' : null}}>{{__('Kazakh')}}</option>
										<option value="ko-KR" {{$settings_two->stablediffusion_default_language == 'ko-KR' ? 'selected' : null}}>{{__('Korean (South Korea)')}}</option>
										<option value="ms-MY" {{$settings_two->stablediffusion_default_language == 'ms-MY' ? 'selected' : null}}>{{__('Malay (Malaysia)')}}</option>
										<option value="nb-NO" {{$settings_two->stablediffusion_default_language == 'nb-NO' ? 'selected' : null}}>{{__('Norwegian (Norway)')}}</option>
										<option value="pl-PL" {{$settings_two->stablediffusion_default_language == 'pl-PL' ? 'selected' : null}}>{{__('Polish (Poland)')}}</option>
										<option value="pt-BR" {{$settings_two->stablediffusion_default_language == 'pt-BR' ? 'selected' : null}}>{{__('Portuguese (Brazil)')}}</option>
										<option value="pt-PT" {{$settings_two->stablediffusion_default_language == 'pt-PT' ? 'selected' : null}}>{{__('Portuguese (Portugal)')}}</option>
										<option value="ro-RO" {{$settings_two->stablediffusion_default_language == 'ro-RO' ? 'selected' : null}}>{{__('Romanian (Romania)')}}</option>
										<option value="ru-RU" {{$settings_two->stablediffusion_default_language == 'ru-RU' ? 'selected' : null}}>{{__('Russian (Russia)')}}</option>
										<option value="sl-SI" {{$settings_two->stablediffusion_default_language == 'sl-SI' ? 'selected' : null}}>{{__('Slovenian (Slovenia)')}}</option>
										<option value="es-ES" {{$settings_two->stablediffusion_default_language == 'es-ES' ? 'selected' : null}}>{{__('Spanish (Spain)')}}</option>
										<option value="sw-KE" {{$settings_two->stablediffusion_default_language == 'sw-KE' ? 'selected' : null}}>{{__('Swahili (Kenya)')}}</option>
										<option value="sv-SE" {{$settings_two->stablediffusion_default_language == 'sv-SE' ? 'selected' : null}}>{{__('Swedish (Sweden)')}}</option>
										<option value="tr-TR" {{$settings_two->stablediffusion_default_language == 'tr-TR' ? 'selected' : null}}>{{__('Turkish (Turkey)')}}</option>
										<option value="vi-VN" {{$settings_two->stablediffusion_default_language == 'vi-VN' ? 'selected' : null}}>{{__('Vietnamese (Vietnam)')}}</option>
									</select>
								</div>
							</div> --}}

                        </div>
                        <button form="settings_form" id="settings_button" class="btn btn-primary w-100">
                            {{ __('Save') }}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>
@endsection

@push('script')
    <script>
        function plagiarismSettingsSave() {
            "use strict";

            document.getElementById("settings_button").disabled = true;
            document.getElementById("settings_button").innerHTML = magicai_localize.please_wait;

            var formData = new FormData();
            formData.append('plagiarism_api_key', $("#plagiarism_api_key").val());
            // formData.append( 'stablediffusion_default_language', $( "#stablediffusion_default_language" ).val() );
            // formData.append( 'stablediffusion_default_model', $( "#stablediffusion_default_model" ).val() );

            $.ajax({
                type: "post",
                url: "/dashboard/admin/settings/plagiarismapi-save",
                data: formData,
                contentType: false,
                processData: false,
                success: function(data) {
                    toastr.success('Settings saved succesfully')
                    document.getElementById("settings_button").disabled = false;
                    document.getElementById("settings_button").innerHTML = "Save";
                },
                error: function(data) {
                    var err = data.responseJSON.errors;
                    $.each(err, function(index, value) {
                        toastr.error(value);
                    });
                    document.getElementById("settings_button").disabled = false;
                    document.getElementById("settings_button").innerHTML = "Save";
                }
            });
            return false;
        }
    </script>
    <script src="/assets/select2/select2.min.js"></script>
@endpush
