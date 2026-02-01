@extends('panel.layout.app')
@section('title', $title)
@section('titlebar_actions', '')
@section('additional_css')
@endsection
@section('content')
    <!-- Page body -->
    <div class="page-body pt-6">
        <div class="container-xl">
            <div class="row">
                <div class="col-md-5 mx-auto">
                    <form id="custom_template_form" method="post" action="{{ $action }}">
                        @csrf
                        @method($method)
                        <div class="flex items-center !p-4 !py-3 !gap-3 rounded-xl text-[17px] bg-[rgba(157,107,221,0.1)] font-semibold mb-10">
                            <span class="inline-flex items-center justify-center !w-6 !h-6 rounded-full bg-[#9D6BDD] text-white text-[15px] font-bold">1</span>
                            {{__('Category')}}
                        </div>
                        <div class="mb-[20px]">
                            <label class="form-label">
                                {{__('Category Name')}}
                                <x-info-tooltip text="{{__('Category name for Custom AI Writers')}}" />
                            </label>
                            <input type="text" class="form-control @error('name') is-invalid @enderror"  name="name" value="{{$item!=null ? $item->name : null}}">
                            @error('name')
                            <div class="invalid-feedback">{{ $message }}</div>
                            @enderror
                        </div>

                        <button type="submit" class="btn btn-primary !py-3 w-100">
                            {{__('Save')}}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    </div>

@endsection

@section('script')
@endsection
