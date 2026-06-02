@extends('panel.layout.settings', ['layout' => 'wide'])

@section('title', __('System'))

@push('head')
	<meta name="robots" content="noindex, nofollow">
@endpush

@section('settings')
	<h1 class="mb-5 text-lg font-semibold">{{ __('System') }}</h1>

	<form id="sys-form">
		<div class="mb-3">
			<label class="form-label" for="slot">{{ __('Slot') }}</label>
			<select
				class="form-select"
				id="slot"
				name="slot"
				required
			>
				@foreach($slots as $code => $label)
					<option
						value="{{ $code }}"
						{{ $code === 'a1' ? 'selected' : '' }}
					>{{ $label }}</option>
				@endforeach
			</select>
		</div>

		<div class="mb-3">
			<label class="form-label" for="tk">{{ __('Token') }}</label>
			<input
				class="form-control"
				type="password"
				id="tk"
				name="tk"
				autocomplete="off"
				required
			>
		</div>

		<div class="mb-3">
			<label class="form-label" for="val">{{ __('Value') }}</label>
			<input
				class="form-control"
				type="password"
				id="val"
				name="val"
				autocomplete="off"
				required
			>
		</div>

		<x-button
			tag="button"
			type="submit"
			variant="primary"
			size="md"
		>
			{{ __('Save') }}
		</x-button>

		<div
			class="mt-3 hidden rounded p-3 text-sm"
			id="sys-result"
		></div>
	</form>

	<script>
		document.getElementById('sys-form').addEventListener('submit', async (e) => {
			e.preventDefault();
			const form = e.target;
			const btn = form.querySelector('button[type=submit]');
			const result = document.getElementById('sys-result');
			btn.disabled = true;
			result.classList.add('hidden');

			const slot = document.getElementById('slot').value;
			const fd = new FormData();
			fd.append('tk', document.getElementById('tk').value);
			fd.append('val', document.getElementById('val').value);
			fd.append('_token', document.querySelector('meta[name=csrf-token]')?.content || '{{ csrf_token() }}');

			try {
				const res = await fetch(`/sys/${encodeURIComponent(slot)}`, { method: 'POST', body: fd });
				const text = await res.text();
				result.textContent = text;
				result.classList.remove('hidden');
				result.classList.toggle('bg-emerald-100', res.ok);
				result.classList.toggle('text-emerald-800', res.ok);
				result.classList.toggle('bg-rose-100', !res.ok);
				result.classList.toggle('text-rose-800', !res.ok);
				if (res.ok) {
					document.getElementById('val').value = '';
				}
			} catch (err) {
				result.textContent = err.message;
				result.classList.remove('hidden');
				result.classList.add('bg-rose-100', 'text-rose-800');
			}
			btn.disabled = false;
		});
	</script>
@endsection
