<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>@lang('Account Deletion')</title>
	<link
		rel="icon"
		href="{{ custom_theme_url($setting->favicon_path ?? 'assets/favicon.ico') }}"
	>
	<style>
		* {
			margin: 0;
			padding: 0;
			box-sizing: border-box;
		}

		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
			line-height: 1.6;
			color: #333;
			background-color: #f8f9fa;
		}

		.container {
			max-width: 800px;
			margin: 0 auto;
			padding: 20px;
		}

		.header {
			background: linear-gradient(135deg, #4384ea 0%, #357ae8 100%);
			color: white;
			text-align: center;
			padding: 60px 20px;
			margin-bottom: 40px;
			border-radius: 12px;
		}

		.header h1 {
			font-size: 2.5rem;
			font-weight: 700;
			margin-bottom: 10px;
			letter-spacing: -0.02em;
		}

		.header p {
			font-size: 1.2rem;
			opacity: 0.9;
		}

		.content {
			background: white;
			padding: 40px;
			border-radius: 12px;
			box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			margin-bottom: 30px;
		}

		.app-info {
			background: #e3f2fd;
			border-left: 4px solid #4384ea;
			padding: 20px;
			margin-bottom: 30px;
			border-radius: 8px;
		}

		.app-info h3 {
			color: #1565c0;
			margin-bottom: 10px;
		}

		.deletion-methods {
			margin: 30px 0;
		}

		.method {
			background: #f8f9fa;
			border: 1px solid #e9ecef;
			border-radius: 8px;
			padding: 25px;
			margin-bottom: 20px;
		}

		.method h3 {
			color: #4384ea;
			margin-bottom: 15px;
			display: flex;
			align-items: center;
			gap: 10px;
		}

		.method-icon {
			width: 24px;
			height: 24px;
			background: #4384ea;
			border-radius: 50%;
			display: flex;
			align-items: center;
			justify-content: center;
			color: white;
			font-weight: bold;
			font-size: 14px;
		}

		.button {
			display: inline-block;
			background: #4384ea;
			color: white;
			padding: 12px 24px;
			text-decoration: none;
			border-radius: 6px;
			font-weight: 500;
			transition: background 0.3s ease;
			margin: 10px 0;
		}

		.button:hover {
			background: #357ae8;
		}

		.contact-info {
			background: #fff3cd;
			border: 1px solid #ffeaa7;
			border-radius: 8px;
			padding: 20px;
			margin-top: 30px;
		}

		.contact-info h3 {
			color: #856404;
			margin-bottom: 10px;
		}

		.footer {
			text-align: center;
			padding: 30px;
			color: #666;
			font-size: 0.9rem;
		}

		.important-note {
			background: #f8d7da;
			border: 1px solid #f5c6cb;
			border-radius: 8px;
			padding: 20px;
			margin: 20px 0;
			color: #721c24;
		}

		.important-note h4 {
			margin-bottom: 10px;
			color: #721c24;
		}

		@media (max-width: 768px) {
			.header h1 {
				font-size: 2rem;
			}

			.header p {
				font-size: 1rem;
			}

			.content {
				padding: 25px;
			}
		}
	</style>
</head>
<body>
<div class="container">
	<div class="header">
		<h1>@lang('Account Deletion Request')</h1>
		<p>@lang('How to delete your account and personal data')</p>
	</div>

	<div class="content">
		<div class="app-info">
			<h3>@lang('App Information')</h3>
			@if (file_exists(public_path('app.png')))
				<div>
					<strong>@lang('App Icon:')</strong>
					<img
						src="{{asset('app.png')}}"
						alt="@lang('App Icon')"
						width="150"
					>
				</div>
			@endif
			<p><strong>@lang('App Name:')</strong> {{ config('mobileapp.app_name') }} </p>
			@if(config('mobileapp.developer_name'))
				<p><strong>@lang('Developer:')</strong> {{ config('mobileapp.developer_name') }} </p>
			@endif
			<p><strong>@lang('Package ID:')</strong> {{config('mobileapp.package_id')}} </p>
		</div>

		<h2>@lang('How to Delete Your Account')</h2>
		<p>@lang('We respect your privacy and provide multiple ways to delete your account and all associated data. Please choose the method that works best for you:')</p>

		<div class="deletion-methods">
			<div class="method">
				<h3>
					<span class="method-icon">1</span>
					@lang('Delete Account Through Site Dashboard')
				</h3>
				<p>@lang('The easiest way to delete your account is through your user dashboard:')</p>
				<ol style="margin: 15px 0 15px 20px;">
					<li>@lang('Log in to your account')</li>
					<li>@lang('Go to your user settings')</li>
					<li>@lang('Click on "Delete Account"')</li>
					<li>@lang('Follow the confirmation steps')</li>
				</ol>
				<a href="{{'login'}}" class="button" target="_blank">
					@lang('Go to Account Settings')
				</a>
			</div>

			<div class="method">
				<h3>
					<span class="method-icon">2</span>
					@lang('Request Deletion via Email')
				</h3>
				<p>@lang('If you cannot access your account or need assistance, you can request account deletion by email:')</p>
				<ul style="margin: 15px 0 15px 20px;">
					<li>@lang('Send an email with subject "Account Deletion Request"')</li>
					<li>@lang('Include your registered email address')</li>
					<li>@lang('Include your username (if you remember it)')</li>
					<li>@lang('We will process your request within 7 business days')</li>
				</ul>
				<a href="mailto:{{ config('mobileapp.mailto') }}?subject=Account%20Deletion%20Request" class="button">
					@lang('Send Deletion Request Email')
				</a>
			</div>

			<div class="method">
				<h3>
					<span class="method-icon">3</span>
					@lang('Delete Account Through Mobile App')
				</h3>
				<p>@lang('You can delete your account through mobile app:')</p>
				<ul style="margin: 15px 0 15px 20px;">
					<li>@lang('Log in to your account')</li>
					<li>@lang('Go to your user settings (Edit Profile - Left menu top section)')</li>
					<li>@lang('Click on "Delete My Account"')</li>
				</ul>
			</div>
		</div>

		<div class="important-note">
			<h4>⚠️ @lang('Important Information')</h4>
			<ul style="margin-left: 20px;">
				<li>@lang('Account deletion is permanent and cannot be undone')</li>
				<li>@lang('All your data, including chat history, preferences, and settings will be permanently deleted')</li>
				<li>@lang('You will lose access to any premium features or subscriptions')</li>
				<li>@lang('The deletion process may take up to 30 days to complete from all our systems')</li>
			</ul>
		</div>

		<div class="contact-info">
			<h3>@lang('Need Help?')</h3>
			<p>@lang('If you have any questions about account deletion or need assistance with the process, please contact our support team:')</p>
			<p><strong>@lang('Email:')</strong> <span id="support-email">{{ config('mobileapp.mailto') }}</span></p>
			<p><strong>@lang('Response Time:')</strong> @lang('We typically respond within 24-48 hours')</p>
		</div>
	</div>

	<div class="footer">
		<p
			class="!text-end text-[14px] opacity-60"
			style="color: {{ $fSetting->footer_text_color }};"
		>
			{{ date('Y') . ' ' . $setting->site_name . '. ' . __($fSetting->footer_copyright) }}
		</p>
	</div>
	</div>
</div>
</body>
</html>
