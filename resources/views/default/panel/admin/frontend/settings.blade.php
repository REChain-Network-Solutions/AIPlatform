@extends('panel.layout.settings')
@section('title', __('Frontend Settings'))
@section('titlebar_actions', '')

@section('settings')
    <form
        id="settings_form"
        onsubmit="return frontendSettingsSave();"
        enctype="multipart/form-data"
    >
        <h3 class="mb-[25px] text-[20px]">{{ __('General Settings') }}</h3>
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Site Name') }}</label>
                    <input
                        class="form-control"
                        id="site_name"
                        type="text"
                        name="site_name"
                        value="{{ $setting->site_name }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Site URL') }}</label>
                    <input
                        class="form-control"
                        id="site_url"
                        type="text"
                        name="site_url"
                        value="{{ $setting->site_url }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Site Email') }}</label>
                    <input
                        class="form-control"
                        id="site_email"
                        type="text"
                        name="site_email"
                        value="{{ $setting->site_email }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Registration Active') }}</label>
                    <select
                        class="form-select"
                        id="register_active"
                        name="register_active"
                    >
                        <option
                            value="1"
                            {{ $setting->register_active == 1 ? 'selected' : '' }}
                        >
                            {{ __('Active') }}</option>
                        <option
                            value="0"
                            {{ $setting->register_active == 0 ? 'selected' : '' }}
                        >
                            {{ __('Passive') }}</option>
                    </select>
                </div>
            </div>

			<div class="col-md-12">
				<div class="mb-3">
					<label class="form-label">{{ __('Registration Active') }}</label>
					<select
						class="form-select"
						id="register_active"
						name="register_active"
					>
						<option
							value="1"
							{{ $setting->register_active == 1 ? 'selected' : '' }}
						>
							{{ __('Active') }}</option>
						<option
							value="0"
							{{ $setting->register_active == 0 ? 'selected' : '' }}
						>
							{{ __('Passive') }}</option>
					</select>
				</div>
			</div>

			<div class="col-md-12">
				<div class="mb-3">
					<label class="form-label">{{ __('Facebook domain verification') }}</label>
					<input
						class="form-control"
						id="facebook_domain_verification"
						type="text"
						name="facebook_domain_verification"
						value="{{ setting('facebook_domain_verification', '') }}"
					>
				</div>
			</div>

			<div class="col-md-12">
				<div class="mb-3">
					<label class="form-label">{{ __('Google No Index') }}</label>
					<select
						class="form-select"
						id="google_robots"
						name="google_robots"
					>
						<option
							value="1"
							{{ setting('google_robots', '0') == 1 ? 'selected' : '' }}
						>
							{{ __('Active') }}</option>
						<option
							value="0"
							{{ setting('google_robots', '0') == 0 ? 'selected' : '' }}
						>
							{{ __('Passive') }}</option>
					</select>
				</div>
			</div>
        </div>

        <div class="row mb-4">
            <h3 class="mb-[25px] text-[20px]">{{ __('Frontend Settings') }}</h3>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('PreHeader Section') }}</label>
                    <select
                        class="form-select"
                        id="preheader_active"
                        name="preheader_active"
                    >
                        <option
                            value="1"
                            {{ $fSectSettings->preheader_active == 1 ? 'selected' : '' }}
                        >
                            {{ __('Active') }}</option>
                        <option
                            value="0"
                            {{ $fSectSettings->preheader_active == 0 ? 'selected' : '' }}
                        >
                            {{ __('Passive') }}</option>
                    </select>
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('PreHeader Title') }}
                    </label>
                    <input
                        class="form-control"
                        id="header_title"
                        type="text"
                        name="header_title"
                        value="{{ $fSetting->header_title }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('PreHeader Text') }}

                    </label>
                    <input
                        class="form-control"
                        id="header_text"
                        type="text"
                        name="header_text"
                        value="{{ $fSetting->header_text }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Sign In Text') }}

                    </label>
                    <input
                        class="form-control"
                        id="sign_in"
                        type="text"
                        name="sign_in"
                        value="{{ $fSetting->sign_in }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Sign Up Text') }}

                    </label>
                    <input
                        class="form-control"
                        id="join_hub"
                        type="text"
                        name="join_hub"
                        value="{{ $fSetting->join_hub }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Hero Subtitle') }}

                    </label>
                    <input
                        class="form-control"
                        id="hero_subtitle"
                        type="text"
                        name="hero_subtitle"
                        value="{{ $fSetting->hero_subtitle }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Hero Title') }}

                    </label>
                    <input
                        class="form-control"
                        id="hero_title"
                        type="text"
                        name="hero_title"
                        value="{{ $fSetting->hero_title }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Hero Title Text Rotator') }}</label>
                    <input
                        class="form-control"
                        id="hero_title_text_rotator"
                        type="text"
                        name="hero_title_text_rotator"
                        value="{{ $fSetting->hero_title_text_rotator }}"
                    >
                    <x-alert class="mt-2">
                        <p>
                            {{ __('Please use comma seperated like; Generator,Chatbot,Assistant') }}
                        </p>
                    </x-alert>
                </div>

            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Hero Description') }}

                    </label>
                    <input
                        class="form-control"
                        id="hero_description"
                        type="text"
                        name="hero_description"
                        value="{{ $fSetting->hero_description }}"
                    >
                </div>
            </div>
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">
                        {{ __('Hero Scroll Text') }}

                    </label>
                    <input
                        class="form-control"
                        id="hero_scroll_text"
                        type="text"
                        name="hero_scroll_text"
                        value="{{ $fSetting->hero_scroll_text }}"
                    >
                </div>
            </div>
            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Hero Button') }}</label>
                    <input
                        class="form-control"
                        id="hero_button"
                        type="text"
                        name="hero_button"
                        value="{{ $fSetting->hero_button }}"
                    >
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Hero Button Type') }}<x-info-tooltip text="{{ __('This will affect the button style') }}" /></label>

                    <select
                        class="form-select"
                        id="hero_button_type"
                        name="hero_button_type"
                    >
                        <option
                            value="1"
                            {{ $fSetting->hero_button_type == 1 ? 'selected' : '' }}
                        >
                            {{ __('Website') }}</option>
                        <option
                            value="0"
                            {{ $fSetting->hero_button_type == 0 ? 'selected' : '' }}
                        >
                            {{ __('Video') }}</option>
                    </select>
                </div>
            </div>

            <div class="col-md-12">
                <div class="mb-3">
                    <label class="form-label">{{ __('Hero Button URL') }}</label>
                    <input
                        class="form-control"
                        id="hero_button_url"
                        type="text"
                        name="hero_button_url"
                        value="{{ $fSetting->hero_button_url }}"
                    >
                </div>
            </div>



            <div class="col-md-12">
                <div class="mb-">
                    <div class="group relative w-full">
                        <button
                            class="mb-3 flex w-full items-center gap-5 text-sm text-heading-foreground"
                            type="button"
                        >
                            <span
                                class="size-12 inline-grid place-items-center rounded-full bg-foreground/[7%] text-heading-foreground transition-colors group-hover:bg-heading-foreground group-hover:text-heading-background"
                            >
                                <x-tabler-plus />
                            </span>
                            @lang('Hero Image')
                        </button>
                        <x-forms.input
                            class="absolute inset-0 z-2 h-full w-full cursor-pointer opacity-0"
                            class:label="leading-tight text-foreground/30"
                            id="hero_image"
                            container-class="static max-w-[270px] mx-auto"
                            size="lg"
                            name="hero_image"
                            type="file"
                            placeholder="{{ __('Upload Hero Image') }}"
                        />
                    </div>
                    @if ($fSetting->hero_image)
                        <div class="my-2">
                            <img
                                class="w-full rounded-lg object-cover"
                                src="{{ $fSetting->hero_image }}"
                                alt="hero image"
                            >
                        </div>
                    @endif

					@if(setting('front_theme') === 'social-media-front')
						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">{{ __('No credit cart required') }}</label>
								<input
									class="form-control"
									id="no_credit_cart_required"
									type="text"
									name="no_credit_cart_required"
									value="{{ $fSetting->no_credit_cart_required }}"
								>
							</div>
						</div>

						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">
									{{ __('Faster content creation') }}
								</label>
								<textarea
									class="form-control"
									id="faster_content_creation"
									name="faster_content_creation"
								>{!! $fSetting->faster_content_creation !!}</textarea>
							</div>
						</div>
						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">
									{{ __('Over 5000 businesses') }}
								</label>
								<textarea
									class="form-control"
									id="over_5000_businesses"
									name="over_5000_businesses"
								>{!! $fSetting->over_5000_businesses !!}</textarea>
							</div>
						</div>
					@endif


					<div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Footer Header') }}

                            </label>
                            <input
                                class="form-control"
                                id="footer_header"
                                type="text"
                                name="footer_header"
                                value="{{ $fSetting->footer_header }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Footer Header Small Text') }}

                            </label>
                            <input
                                class="form-control"
                                id="footer_text_small"
                                type="text"
                                name="footer_text_small"
                                value="{{ $fSetting->footer_text_small }}"
                            >
                        </div>
                    </div>
                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Footer Text') }}

                            </label>
                            <input
                                class="form-control"
                                id="footer_text"
                                type="text"
                                name="footer_text"
                                value="{{ $fSetting->footer_text }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Footer Button Text') }}
                            </label>
                            <input
                                class="form-control"
                                id="footer_button_text"
                                type="text"
                                name="footer_button_text"
                                value="{{ $fSetting->footer_button_text }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Footer Button URL (Please enter full url)') }}</label>
                            <input
                                class="form-control"
                                id="footer_button_url"
                                type="text"
                                name="footer_button_url"
                                value="{{ $fSetting->footer_button_url }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Footer Copyright') }}
                            </label>
                            <input
                                class="form-control"
                                id="footer_copyright"
                                type="text"
                                name="footer_copyright"
                                value="{{ $fSetting->footer_copyright }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <x-forms.input
                                id="footer_text_color"
                                name="footer_text_color"
                                label="{{ __('Footer Text Color') }}"
                                size="lg"
                                type="color"
                                value="{{ $fSetting->footer_text_color }}"
                                tooltip="{{ __('Pick a color for for the icon container shape. Color is in HEX format.') }}"
                            />
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Pricing Section') }}
                            </label>
                            <select
                                class="form-select"
                                id="frontend_pricing_section"
                                name="frontend_pricing_section"
                            >
                                <option
                                    value="1"
                                    {{ $setting->frontend_pricing_section == 1 ? 'selected' : '' }}
                                >
                                    {{ __('Active') }}</option>
                                <option
                                    value="0"
                                    {{ $setting->frontend_pricing_section == 0 ? 'selected' : '' }}
                                >
                                    {{ __('Passive') }}</option>
                            </select>
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Custom Templates Section') }}</label>
                            <select
                                class="form-select"
                                id="frontend_custom_templates_section"
                                name="frontend_custom_templates_section"
                            >
                                <option
                                    value="1"
                                    {{ $setting->frontend_custom_templates_section == 1 ? 'selected' : '' }}
                                >
                                    {{ __('Active') }}</option>
                                <option
                                    value="0"
                                    {{ $setting->frontend_custom_templates_section == 0 ? 'selected' : '' }}
                                >
                                    {{ __('Passive') }}</option>
                            </select>
                        </div>
                    </div>

                </div>

                <div class="row mb-4">
                    <h3 class="mb-[25px] text-[20px]">{{ __('Floating Button') }}</h3>
                    <div class="col-md-12">
                        <div class="mb-3">
                            <select
                                class="form-select"
                                id="floating_button_active"
                                name="floating_button_active"
                            >
                                <option
                                    value="1"
                                    {{ $fSetting->floating_button_active == 1 ? 'selected' : '' }}
                                >
                                    {{ __('Active') }}</option>
                                <option
                                    value="0"
                                    {{ $fSetting->floating_button_active == 0 ? 'selected' : '' }}
                                >
                                    {{ __('Passive') }}</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-md-12 floating-button-input">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Floating Button Small Text') }}</label>
                            <input
                                class="form-control"
                                id="floating_button_small_text"
                                type="text"
                                name="floating_button_small_text"
                                value="{{ $fSetting->floating_button_small_text }}"
                            >
                        </div>
                    </div>
                    <div class="col-md-12 floating-button-input">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Floating Button Bold Text') }}</label>
                            <input
                                class="form-control"
                                id="floating_button_bold_text"
                                type="text"
                                name="floating_button_bold_text"
                                value="{{ $fSetting->floating_button_bold_text }}"
                            >
                        </div>
                    </div>
                    <div class="col-md-12 floating-button-input">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Floating Button URL') }}</label>
                            <input
                                class="form-control"
                                id="floating_button_link"
                                type="text"
                                name="floating_button_link"
                                value="{{ $fSetting->floating_button_link }}"
                            >
                        </div>
                    </div>

                    <div class="floating-button-input mt-2 flex items-center justify-center">
                        <a
                            class="flex max-w-max items-center gap-3 rounded-xl bg-white px-3 py-2 text-sm text-[#002A40] text-opacity-60 shadow-lg transition-all duration-300 hover:-translate-y-1 hover:scale-110 hover:no-underline hover:shadow-md"
                            id="floating_button_preview"
                            data-fslightbox="html5-youtube-videos"
                            href="{{ !empty($fSetting->floating_button_link) ? $fSetting->floating_button_link : '#' }}"
                        >
                            <span
                                class="lqd-is-in-view inline-flex shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-[#3655df] via-[#A068FA] via-70% to-[#327BD1]"
                            >
                                <svg
                                    style="padding: 16px;"
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="45"
                                    height="45"
                                    viewBox="0 0 24 24"
                                    stroke-width="2"
                                    stroke="currentColor"
                                    fill="none"
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                >
                                    <path
                                        stroke="none"
                                        d="M0 0h24v24H0z"
                                        fill="none"
                                    ></path>
                                    <path
                                        d="M6 4v16a1 1 0 0 0 1.524 .852l13 -8a1 1 0 0 0 0 -1.704l-13 -8a1 1 0 0 0 -1.524 .852z"
                                        stroke-width="0"
                                        fill="#fff"
                                    ></path>
                                </svg>
                            </span>
                            <p class="[&amp;_strong]:block pt-2">{!! __($fSetting->floating_button_small_text ?? 'See it in action') !!}<strong class="text-[0.9rem] text-black">{!! __($fSetting->floating_button_bold_text ?? 'How it Works?') !!} &nbsp;</strong></p>
                        </a>
                    </div>
                </div>

                <div class="row mb-4">
                    <h3 class="mb-[25px] text-[20px]">{{ __('Footer Social Media Settings') }}</h3>
                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">Facebook {{ __('Address') }}</label>
                            <input
                                class="form-control"
                                id="frontend_footer_facebook"
                                type="text"
                                name="frontend_footer_facebook"
                                value="{{ $setting->frontend_footer_facebook }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">Twitter {{ __('Address') }}</label>
                            <input
                                class="form-control"
                                id="frontend_footer_twitter"
                                type="text"
                                name="frontend_footer_twitter"
                                value="{{ $setting->frontend_footer_twitter }}"
                            >
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">Instagram {{ __('Address') }}</label>
                            <input
                                class="form-control"
                                id="frontend_footer_instagram"
                                type="text"
                                name="frontend_footer_instagram"
                                value="{{ $setting->frontend_footer_instagram }}"
                            >
                        </div>
                    </div>

                </div>

				@if(setting('front_theme') === 'marketing-bot')
					<div class="row mb-4">
						<h3 class="mb-[25px] text-[20px]">{{ __('Footer Use Cases Settings') }}</h3>

						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">{{ __('Footer Use Cases Links') }}</label>
								<div id="use-cases-container">
									@php
										$defaultUseCaseLinks = [
											['link' => '#', 'title' => 'Collect Feedback'],
											['link' => '#', 'title' => 'Product Launches'],
											['link' => '#', 'title' => 'Promotions'],
											['link' => '#', 'title' => 'Limited Offers'],
											['link' => '#', 'title' => 'Reminders'],
											['link' => '#', 'title' => 'Events'],
											['link' => '#', 'title' => 'Bulk Messages'],
										];
										$useCaseLinks = json_decode(setting('footer_use_cases_links', ''), true);
										if (!is_array($useCaseLinks) || empty($useCaseLinks)) {
											$useCaseLinks = $defaultUseCaseLinks;
										}
									@endphp
									@foreach ($useCaseLinks as $index => $link)
										<div class="use-case-item mb-3 p-3 border rounded">
											<div class="row">
												<div class="col-md-5">
													<label class="form-label">{{ __('Title') }}</label>
													<input
														class="form-control"
														type="text"
														name="use_cases[{{ $index }}][title]"
														value="{{ $link['title'] ?? '' }}"
														placeholder="{{ __('Enter link title') }}"
													>
												</div>
												<div class="col-md-5">
													<label class="form-label">{{ __('Link') }}</label>
													<input
														class="form-control"
														type="text"
														name="use_cases[{{ $index }}][link]"
														value="{{ $link['link'] ?? '' }}"
														placeholder="{{ __('Enter link URL') }}"
													>
												</div>
												<div class="col-md-2 d-flex align-items-end">
													<button
														type="button"
														class="btn btn-danger btn-sm remove-use-case"
														onclick="removeUseCaseItem(this)"
													>
														<i class="fa fa-trash"></i>
													</button>
												</div>
											</div>
										</div>
									@endforeach
								</div>
								<button
									type="button"
									class="btn btn-secondary btn-sm mt-2"
									onclick="addUseCaseItem()"
								>
									<i class="fa fa-plus"></i> {{ __('Add New Use Case Link') }}
								</button>
							</div>
						</div>
					</div>

					<div class="row mb-4">
						<h3 class="mb-[25px] text-[20px]">{{ __('Footer Resources Settings') }}</h3>

						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">{{ __('Footer Resources Links') }}</label>
								<div id="resources-container">
									@php
										$defaultResourceLinks = [
											['link' => '#', 'title' => 'Terms of Services'],
											['link' => '#', 'title' => 'Support'],
											['link' => '#', 'title' => 'Help Center'],
											['link' => '#', 'title' => 'Support'],
											['link' => '#', 'title' => 'Privacy Policy'],
											['link' => '#', 'title' => 'Blog'],
											['link' => '#', 'title' => 'Status'],
										];
										$resourceLinks = json_decode(setting('footer_resources_links', ''), true);
										if (!is_array($resourceLinks) || empty($resourceLinks)) {
											$resourceLinks = $defaultResourceLinks;
										}
									@endphp
									@foreach ($resourceLinks as $index => $link)
										<div class="resource-item mb-3 p-3 border rounded">
											<div class="row">
												<div class="col-md-5">
													<label class="form-label">{{ __('Title') }}</label>
													<input
														class="form-control"
														type="text"
														name="resources[{{ $index }}][title]"
														value="{{ $link['title'] ?? '' }}"
														placeholder="{{ __('Enter link title') }}"
													>
												</div>
												<div class="col-md-5">
													<label class="form-label">{{ __('Link') }}</label>
													<input
														class="form-control"
														type="text"
														name="resources[{{ $index }}][link]"
														value="{{ $link['link'] ?? '' }}"
														placeholder="{{ __('Enter link URL') }}"
													>
												</div>
												<div class="col-md-2 d-flex align-items-end">
													<button
														type="button"
														class="btn btn-danger btn-sm remove-resource"
														onclick="removeResourceItem(this)"
													>
														<i class="fa fa-trash"></i>
													</button>
												</div>
											</div>
										</div>
									@endforeach
								</div>
								<button
									type="button"
									class="btn btn-secondary btn-sm mt-2"
									onclick="addResourceItem()"
								>
									<i class="fa fa-plus"></i> {{ __('Add New Resource Link') }}
								</button>
							</div>
						</div>
					</div>

					<script>
						let useCaseIndex = {{ count($useCaseLinks ?? []) }};
						let resourceIndex = {{ count($resourceLinks ?? []) }};

						function addUseCaseItem() {
							const container = document.getElementById('use-cases-container');
							const newItem = document.createElement('div');
							newItem.className = 'use-case-item mb-3 p-3 border rounded';
							newItem.innerHTML = `
				<div class="row">
					<div class="col-md-5">
						<label class="form-label">{{ __('Title') }}</label>
						<input
							class="form-control"
							type="text"
							name="use_cases[${useCaseIndex}][title]"
							value=""
							placeholder="{{ __('Enter link title') }}"
						>
					</div>
					<div class="col-md-5">
						<label class="form-label">{{ __('Link') }}</label>
						<input
							class="form-control"
							type="text"
							name="use_cases[${useCaseIndex}][link]"
							value=""
							placeholder="{{ __('Enter link URL') }}"
						>
					</div>
					<div class="col-md-2 d-flex align-items-end">
						<button
							type="button"
							class="btn btn-danger btn-sm remove-use-case"
							onclick="removeUseCaseItem(this)"
						>
							<i class="fa fa-trash"></i>
						</button>
					</div>
				</div>
			`;
							container.appendChild(newItem);
							useCaseIndex++;
						}

						function removeUseCaseItem(button) {
							const item = button.closest('.use-case-item');
							item.remove();
						}

						function addResourceItem() {
							const container = document.getElementById('resources-container');
							const newItem = document.createElement('div');
							newItem.className = 'resource-item mb-3 p-3 border rounded';
							newItem.innerHTML = `
				<div class="row">
					<div class="col-md-5">
						<label class="form-label">{{ __('Title') }}</label>
						<input
							class="form-control"
							type="text"
							name="resources[${resourceIndex}][title]"
							value=""
							placeholder="{{ __('Enter link title') }}"
						>
					</div>
					<div class="col-md-5">
						<label class="form-label">{{ __('Link') }}</label>
						<input
							class="form-control"
							type="text"
							name="resources[${resourceIndex}][link]"
							value=""
							placeholder="{{ __('Enter link URL') }}"
						>
					</div>
					<div class="col-md-2 d-flex align-items-end">
						<button
							type="button"
							class="btn btn-danger btn-sm remove-resource"
							onclick="removeResourceItem(this)"
						>
							<i class="fa fa-trash"></i>
						</button>
					</div>
				</div>
			`;
							container.appendChild(newItem);
							resourceIndex++;
						}

						function removeResourceItem(button) {
							const item = button.closest('.resource-item');
							item.remove();
						}
					</script>
				@endif

                <div class="row mb-4">
                    <h3 class="mb-[25px] text-[20px]">{{ __('Advanced Settings') }}</h3>
					<div class="col-md-12">
						<div class="mb-3" x-data="{ selectedType: '{{ setting('frontend_additional_url_type', 'default') }}' }">
							<x-forms.input
								id="frontend_additional_url_type"
								name="frontend_additional_url_type"
								type="select"
								label="{{ __('Landing Page') }}"
								tooltip="{{ __('Choose which page visitors see when they access your site') }}"
								x-model="selectedType"
							>
								<option value="default" :selected="selectedType === 'default'">
									{{ __('Default Landing Page') }}
								</option>
								@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-chat-pro'))
									<option value="ai-chat-pro" :selected="selectedType === 'ai-chat-pro'">
										{{ __('AI Chat Pro') }}
									</option>
								@endif
								@if(\App\Helpers\Classes\MarketplaceHelper::isRegistered('ai-image-pro'))
									<option value="ai-image-pro" :selected="selectedType === 'ai-image-pro'">
										{{ __('AI Image Pro') }}
									</option>
								@endif
								<option value="custom" :selected="selectedType === 'custom'">
									{{ __('Custom Link') }}
								</option>
							</x-forms.input>
							<div x-show="selectedType === 'custom'" x-cloak x-transition class="mt-3">
								<x-forms.input
									id="frontend_additional_url"
									name="frontend_additional_url"
									type="text"
									label="{{ __('Custom Landing Page URL') }}"
									placeholder="https://example.com/custom-page or /custom-page"
									value="{{ setting('frontend_additional_url') }}"
								/>
							</div>
						</div>
					</div>
                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Custom CSS URL') }}</label>
                            <input
                                class="form-control"
                                id="frontend_custom_css"
                                type="text"
                                name="frontend_custom_css"
                                value="{{ $setting->frontend_custom_css }}"
                            >
                            <x-alert class="!mt-2">
                                <p>
                                    {{ __('Please provide full URL with http:// or https://') }}
                                </p>
                            </x-alert>
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">{{ __('Custom JS URL') }}</label>
                            <input
                                class="form-control"
                                id="frontend_custom_js"
                                type="text"
                                name="frontend_custom_js"
                                value="{{ $setting->frontend_custom_js }}"
                            >
                            <x-alert class="!mt-2">
                                <p>
                                    {{ __('Please provide full URL with http:// or https://') }}
                                </p>
                            </x-alert>
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Code before </head>') }}
                                <x-info-tooltip text="{{ __('Only accepts javascript code wrapped with <script> tags and HTML markup that is valid inside the </head> tag.') }}" />
                            </label>
                            <textarea
                                class="form-control"
                                id="frontend_code_before_head"
                                name="frontend_code_before_head"
                            >{{ $setting->frontend_code_before_head }}</textarea>
                        </div>
                    </div>

                    <div class="col-md-12">
                        <div class="mb-3">
                            <label class="form-label">
                                {{ __('Code before </body>') }}
                                <x-info-tooltip text="{{ __('Only accepts javascript code wrapped with <script> tags and HTML markup that is valid inside the </body> tag.') }}" />
                            </label>
                            <textarea
                                class="form-control"
                                id="frontend_code_before_body"
                                name="frontend_code_before_body"
                            >{{ $setting->frontend_code_before_body }}</textarea>
                        </div>
                    </div>

                </div>

                <button
                    class="btn btn-primary w-full"
                    id="settings_button"
                    form="settings_form"
                >
                    {{ __('Save') }}
                </button>
			</div>
		</div>
    </form>
@endsection

@push('script')
    <script src="{{ custom_theme_url('/assets/js/panel/settings.js') }}"></script>
    <script
        src="{{ custom_theme_url('/assets/libs/ace/src-min-noconflict/ace.js') }}"
        type="text/javascript"
        charset="utf-8"
    ></script>
    <style
        type="text/css"
        media="screen"
    >
        .ace_editor {
            min-height: 200px;
        }
    </style>
    <script>
        var frontend_code_before_head = ace.edit("frontend_code_before_head");
        frontend_code_before_head.session.setMode("ace/mode/html");

        var frontend_code_before_body = ace.edit("frontend_code_before_body");
        frontend_code_before_body.session.setMode("ace/mode/html");

		if($('#faster_content_creation')) {
			var faster_content_creation = ace.edit("faster_content_creation");
			frontend_code_before_body.session.setMode("ace/mode/html");
		}

		if($('#over_5000_businesses')){
			var over_5000_businesses = ace.edit("over_5000_businesses");
			over_5000_businesses.session.setMode("ace/mode/html");
		}

	</script>
    <script>
        $(document).ready(function() {
            'use strict';

            if ($('#floating_button_active').val() == '0') {
                $('.floating-button-input').hide();
            }

            $('#floating_button_active').change(function() {
                var selectedValue = $(this).val();
                if (selectedValue == '1') {
                    $('.floating-button-input').show();
                } else {
                    $('.floating-button-input').hide();
                }
            });

            var smallTextInput = $('#floating_button_small_text');
            var boldTextInput = $('#floating_button_bold_text');
            var previewLink = $('#floating_button_preview');

            smallTextInput.on('input', function() {
                var smallText = smallTextInput.val() || "See it in action";
                var boldText = boldTextInput.val() || "How it Works?";
                updatePreview(smallText, boldText);
            });

            boldTextInput.on('input', function() {
                var smallText = smallTextInput.val() || "See it in action";
                var boldText = boldTextInput.val() || "How it Works?";
                updatePreview(smallText, boldText);
            });

            // Function to update the preview <a> tag
            function updatePreview(smallText, boldText) {
                var updatedContent = smallText + '<strong class="text-black text-[0.9rem]">' + boldText +
                    '</strong> &nbsp;';
                previewLink.find('p').html(updatedContent).addClass('pt-4');
            }

        });
    </script>
@endpush
