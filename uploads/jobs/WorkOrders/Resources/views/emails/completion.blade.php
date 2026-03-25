@php($brand = \Modules\WorkOrders\Services\BrandingService::get())
@extends('workorders::emails.layout')
@section('content')
<p>{{ __('workorders::mail.completion.body', ['id'=>$wo->id]) }}</p>
<p><a href="{{ $pdfUrl }}">{{ __('workorders::mail.completion.cta_pdf') }}</a></p>
@endsection