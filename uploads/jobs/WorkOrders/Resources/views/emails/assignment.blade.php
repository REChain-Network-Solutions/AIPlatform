@php($brand = \Modules\WorkOrders\Services\BrandingService::get())
@extends('workorders::emails.layout')
@section('content')
<p>{{ __('workorders::mail.assignment.greeting') }}</p>
<p>{{ __('workorders::mail.assignment.body', ['id'=>$wo->id, 'when'=>$wo->scheduled_at]) }}</p>
<p><a href="{{ $portalUrl }}">{{ __('workorders::mail.assignment.cta') }}</a></p>
@endsection