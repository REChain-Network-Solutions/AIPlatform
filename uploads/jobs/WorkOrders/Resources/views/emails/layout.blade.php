<!doctype html><html><body style='font-family:Arial,Helvetica,sans-serif;font-size:14px;color:#222'>
<div style="max-width:680px;margin:auto">
  @if(!empty($brand['logo']))<p><img src="{{ $brand['logo'] }}" style="max-height:48px"></p>@endif
  <h3 style="margin:0 0 6px 0">{{ $title ?? '' }}</h3>
  <p style="color:#555; margin-top:0">{{ $subtitle ?? '' }}</p>
  <div>@yield('content')</div>
  <hr style="margin:24px 0">
  <div style="color:#888; font-size:12px">{{ $brand['footer'] ?? '' }}</div>
</div></body></html>