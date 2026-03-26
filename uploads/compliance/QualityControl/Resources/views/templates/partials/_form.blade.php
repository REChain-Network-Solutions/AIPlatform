@php($template = $template ?? null)
<div class="mb-3">
  <label class="form-label">{{ __('quality_control::templates.fields.name') }}</label>
  <input type="text" name="name" class="form-control" value="{{ old('name', $template->name ?? '') }}" required>
</div>

<div class="mb-3">
  <label class="form-label">{{ __('quality_control::templates.fields.trade') }}</label>
  <input type="text" name="trade" class="form-control" value="{{ old('trade', $template->trade ?? '') }}">
</div>

<div class="mb-3">
  <label class="form-label">{{ __('quality_control::templates.fields.description') }}</label>
  <textarea name="description" class="form-control" rows="4">{{ old('description', $template->description ?? '') }}</textarea>
</div>

<div class="form-check mb-3">
  <input class="form-check-input" type="checkbox" name="is_active" value="1" id="is_active" {{ old('is_active', $template->is_active ?? true) ? 'checked' : '' }}>
  <label class="form-check-label" for="is_active">{{ __('quality_control::templates.fields.active') }}</label>
</div>
