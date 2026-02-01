@extends('panel.layout.settings')
@section('title', __('AI Music'))
@section('titlebar_subtitle', __('You can generate clip lyrics, a title, and tags based on your short description of the song you want.'))
@section('titlebar_actions', '')
@section('settings')

	<h2 class="mb-4">{{__('Upload or Provide a Link to an Audio File')}}</h2>
	<form id="uploadForm" action="{{ route('dashboard.user.ai-music.store') }}" method="POST" enctype="multipart/form-data">
		@csrf

		<div class="mb-3">
			<label class="form-label">{{__('Upload an MP3 File')}}</label>
			<input type="file" name="audio" class="form-control" accept="audio/*">
			<x-alert class="!mt-2">
				{!!  __('<strong>Tip:</strong> Upload an MP3 file (minimum 10 seconds long) based on your selected purpose:<br><br>
					<ul>
						<li><strong>Song:</strong> Must contain both vocals (singing) and instrumental.</li>
						<li><strong>Voice:</strong> Must contain only vocals (singing, no background music).</li>
						<li><strong>Instrumental:</strong> Must contain only instrumental (no vocals).</li>
					</ul>
					<br>
					For "Song" and "Voice" purposes, vocals must be in singing form. Normal speech is not supported.')  !!}
			</x-alert>
		</div>

		<div class="mb-3">
			<label class="form-label">{{__('Or Provide an MP3 Link')}}</label>
			<input type="url" name="link" class="form-control" placeholder="https://example.com/audio.mp3">
		</div>

		<div class="mb-3">
			<label class="form-label">{{__('Select Purpose')}}</label>
			<select name="purpose" class="form-control" required>
				<option value="song">{{__('Song')}}</option>
				<option value="voice">{{__('Voice')}}</option>
				<option value="instrumental">{{__('Instrumental')}}</option>
			</select>
			<x-alert class="!mt-2">
				{!!  __('<strong>Tip:</strong> Choose how you want to process your audio:<br><br>
					<ul>
						<li><strong>Song:</strong> Extract both vocals and instrumental.</li>
						<li><strong>Voice:</strong> Extract only vocals.</li>
						<li><strong>Instrumental:</strong> Extract only instrumental.</li>
					</ul>
					<br>
					Your output will depend on your selection.')  !!}
			</x-alert>
		</div>

		<div class="mb-3">
			<div x-data="{ exampleFormat: '' }" class="flex flex-col gap-3">
				<div class="flex justify-between">
					<label> {{ __('Lyrics') }} </label>
					<x-button
					class="chat-completions-fill-btn"
					type="button"
					size="sm"
					@click="exampleFormat =
`[Verse]
Silver cities shine brightly
Skies are painted blue
Hopes and dreams take flight
Future starts anew

[Verse 2]
Machines hum a new tune
Worlds we’ve never seen
Chasing stars so far
Building our own dream

[Chorus]
Future dreams so high
Touch the endless sky
Live beyond the now
Make the future wow

[Bridge]
With every beat we rise
See through wiser eyes
The places we can go
A brilliance that will grow`">
						{{ __('Example Lyrics') }}
					</x-button>
				</div>
				<x-forms.input
					id="lyrics_description_prompt"
					name="lyrics"
					size="lg"
					type="textarea"
					rows="10"
					x-model="exampleFormat"
					required
				>
				</x-forms.input>
			</div>
		</div>
		@if ($app_is_demo)
			<x-button
				type="button"
				onclick="return toastr.info('This feature is disabled in Demo version.');"
			>
				{{ __('Generate Song') }}
			</x-button>
		@else
			<x-button
				type="submit"
				form="uploadForm"
				size="lg"
			>
				{{ __('Generate Song') }}
			</x-button>
		@endif
	</form>

	<div id="responseMessage" class="mt-3"></div>
@endsection
