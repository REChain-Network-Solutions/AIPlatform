@php($brand = \Modules\WorkOrders\Services\BrandingService::get())
@extends('workorders::emails.layout')
@section('content')
<p>{{ __('workorders::mail.arrival.body', ['when'=>$wo->scheduled_at]) }}</p>
<p><a href="{{ $portalUrl }}">{{ __('workorders::mail.generic.portal_cta') }}</a></p>
@endsection