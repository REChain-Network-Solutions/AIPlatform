@extends('panel.layout.settings')
@section('title', __('Premium Advantages'))
@section('titlebar_actions', '')
@section('additional_css')
    <link
        rel="stylesheet"
        href="https://foliotek.github.io/Croppie/croppie.css"
    />
    <style>
        #upload-demo {
            width: 250px;
            height: 250px;
            padding-bottom: 25px;
            margin: 0 auto;
        }
    </style>
@endsection

@section('settings')
    <h3 class="mb-[25px] text-[20px]">{{ __('Premium Advantages') }}</h3>
    <form action="{{ route("dashboard.admin.config.premium-advantages.store") }}" method="POST"
          enctype="multipart/form-data">
        <div class="row">
			@foreach($premiumAdvantages as $key => $premiumAdvantage)
				<div class="col-md-12">
					<div class="mb-3">
						<input
							class="form-control @error($key) is-invalid @enderror"
							id="{{ $key }}"
							type="text"
							name="{{ $key }}"
							value="{{ old($key, $premiumAdvantage) }}"
						>
					</div>
				</div>
			@endforeach
        </div>
		<div class="row">
			<div class="col-md-12 mt-4 mx-auto">
				<button
					class="btn btn-primary w-full"
					type="submit"
				>
					{{ __('Save') }}
				</button>
			</div>
		</div>
    </form>
@endsection
@push('script')
@endpush
