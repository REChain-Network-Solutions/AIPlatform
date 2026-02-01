@extends('panel.layout.app')
@section('title', $title)
@section('titlebar_actions', '')
@section('content')
	<!-- Page body -->
	<div class="page-body pt-6">
		<div class="container-xl">
			<div class="row">
				<div class="col-md-5 mx-auto">
					<form
							class="@if(view()->exists('panel.admin.custom.layout.panel.header-top-bar')) bg-[--tblr-bg-surface] px-8 py-10 rounded-[--tblr-border-radius] @endif"
							action="{{ $action }}"
							method="post"
							enctype="multipart/form-data"
					>
						@csrf
						@method($method)

						<div class="flex items-center !p-4 !py-3 !gap-3 rounded-xl text-[17px] bg-[rgba(157,107,221,0.1)] font-semibold mb-10">
							<span class="inline-flex items-center justify-center !w-6 !h-6 rounded-full bg-[#9D6BDD] text-white text-[15px] font-bold">1</span>
							{{__('Template')}}
						</div>

						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Template Name')}}
								<x-info-tooltip text="{{__('Pick a name for the template.')}}" />
							</label>
							<input type="text" class="form-control @error('name') is-invalid @enderror" id="name" name="name" value="{{$template!=null ? $template->name : null}}">
							@error('name')
							<div class="invalid-feedback">
								{{ $message }}
							</div>
							@enderror
						</div>
						<div class="col-md-12">
							<div class="mb-3">
								<label class="form-label">{{__('Category')}}<x-info-tooltip text="{{__('Pick a category for the template.')}}" /></label>
								<select class="form-select" name="chat_category" id="chat_category">
									@if ($template!=null)
										<option value="" {{$template->category == '' ? 'selected' : null}}>{{__('Default')}}</option>
										@foreach ($categoryList as $category)
											<option value="{{$category->name}}" {{$template->category == $category->name ? 'selected' : null}}>{{$category->name}}</option>
										@endforeach
									@else
										<option value="" selected>{{__('Default')}}</option>
										@foreach ($categoryList as $category)
											<option {{ $template->category == $category->name ? 'selected': '' }} value="{{$category->name}}" >{{ $category->name }}</option>
										@endforeach
									@endif
								</select>
							</div>
						</div>
						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Template Short Name')}}
								<x-info-tooltip text="{{__('Shortened name of the template or human name. Maximum 3 letters is suggested.')}}" />
							</label>
							<input type="text" class="form-control" id="short_name" name="short_name" value="{{$template!=null ? $template->short_name : null}}">
						</div>
						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Description')}}
								<x-info-tooltip text="{{__('A short description of what this chat template can help with.')}}" />
							</label>
							<textarea class="form-control" id="description" name="description">{{$template!=null ? $template->description : null}}</textarea>
						</div>

						@if(\Illuminate\Support\Facades\Schema::hasColumn('openai_chat_category', 'first_message'))
							<div class="mb-[20px]">
								<label class="form-label">
									{{__('First Message')}}
								</label>
								<input type="text" class="form-control" id="first_message" name="first_message" value="{{$template!=null ? $template->first_message : null}}">
							</div>
						@endif


						@if(\Illuminate\Support\Facades\Schema::hasColumn('openai_chat_category', 'first_message'))
							<div class="mb-[20px]">
								<label class="form-label">
									{{__('Instructions')}}
									<x-info-tooltip text="{{__('You can provide instructions to your GPT-3 model to ensure it aligns with your brand and tone.')}}" />
								</label>
								<textarea class="form-control" id="instructions" name="instructions">{{$template!=null ? $template->instructions : null}}</textarea>
							</div>
						@endif

						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Avatar')}}
								<x-info-tooltip text="{{__('Avatar will shown in chat page.')}}" />
							</label>
							<input type="file" class="form-control" id="avatar" name="avatar" value="{{$template!=null ? $template->short_name : null}}">
						</div>
						<div class="mb-20">
							<label class="form-label">
								{{__('Template Color')}}
								<x-info-tooltip text="{{__('Pick a color for for the icon container shape. Color is in HEX format.')}}" />
							</label>
							<div class="form-control flex items-center gap-2 relative">
								<div class="w-[17px] h-[17px] rounded-full overflow-hidden">
									<input type="color" class="w-[150%] h-[150%] relative -start-1/4 -top-1/4 p-0 rounded-full border-none cursor-pointer appearance-none" id="color" name="color" value="{{$template!=null ? $template->color : '#8fd2d0'}}">
								</div>
								<input class="bg-transparent border-none outline-none text-inherit" id="color_value" name="color_value" value="{{$template!=null ? $template->color : '#8fd2d0'}}" />
							</div>
						</div>

						<div class="flex items-center !p-4 !py-3 !gap-3 rounded-xl text-[17px] bg-[rgba(157,107,221,0.1)] font-semibold mb-10">
							<span class="inline-flex items-center justify-center !w-6 !h-6 rounded-full bg-[#9D6BDD] text-white text-[15px] font-bold">2</span>
							{{__('Personality')}}
						</div>

						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Human Name')}}
								<x-info-tooltip text="{{__('Define a human name for the chatbot to give it more personality.')}}" />
							</label>
							<input type="text" class="form-control" id="human_name" placeholder="Allison Burgers" name="human_name" value="{{$template!=null ? $template->human_name : null}}">
						</div>
						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Template Role')}}
								<x-info-tooltip text="{{__('A role for the chatbot that can define what it can help with. For example Finance Expert.')}}" />
							</label>
							<input type="text" class="form-control" id="role" name="role" placeholder="Finance Expert" value="{{$template!=null ? $template->role : null}}">
						</div>
						<div class="mb-[20px]">
							<label class="form-label">
								{{__('Helps With')}}
								<x-info-tooltip text="{{__('Describe what this chatbot can help with. It shows when starting a conversation and the chatbot introducing itself.')}}" />
							</label>
							<textarea class="form-control" id="helps_with" placeholder="I can help you with managing your finance" name="helps_with">{{$template!=null ? $template->helps_with : null}}</textarea>
						</div>
						<div class="mb-[20px]">
							<label class="form-label" for="chatbot_id">
								{{__('Chatbot Training')}}
								<x-info-tooltip text="{{__('Choose any trained chatbot. If you need to train a new chatbot, visit the Chatbot Training')}}" />
							</label>
							<select name="chatbot_id" id="chatbot_id" class="form-control">
								<option value="0">Select Chatbot</option>
								@foreach($chatbots as $chatbot)
									<option {{ $template?->chatbot_id == $chatbot->id ? 'selected': '' }} value="{{ $chatbot->id }}" > {{ $chatbot->title }} </option>
								@endforeach
							</select>
						</div>
						<div class="mb-[20px]">
							<label class="form-label" for="assistant">
								{{__('AI Assistants')}}
								<x-info-tooltip text="{{__('Choose your Openai Assistant. If you need to train a new chatbot, visit the Chatbot Training')}}" />
							</label>
							<select name="assistant" id="assistant" class="form-control">
								<option value="">Select Assistant</option>
								@foreach($assistants as $assistant)
									<option {{ $template?->assistant == $assistant["id"] ? 'selected': '' }} value="{{ $assistant["id"] }}" > {{ $assistant["name"] }} </option>
								@endforeach
							</select>
						</div>
						{{--						<div id="chatbot_training_json" class="mb-[20px] {{ is_numeric($template?->chatbot_id) && $template?->chatbot_id != '0' ? 'd-none' : '' }}" >--}}
						{{--							<label class="form-label">--}}
						{{--								{{__('Chatbot Training')}}--}}
						{{--								<x-info-tooltip text="{{__('Chat models take a list of messages as input and return a model-generated message as output. Although the chat format is designed to make multi-turn conversations easy, it’s just as useful for single-turn tasks without any conversation. Add your custom JSON data.')}}" />--}}
						{{--								<button type="button" class="chat-completions-fill-btn bg-primary !bg-opacity-5 border-none !px-3 !py-1 rounded-full transition-transform cursor-pointer text-sm font-medium text-white">{{__('Create example input')}}</button>--}}
						{{--							</label>--}}
						{{--							<textarea class="form-control" id="chat_completions" name="chat_completions">{{$template!=null ? $template->chat_completions : null}}</textarea>--}}
						{{--							<div class="!mt-3">--}}
						{{--								<a class="text-heading" href="https://platform.openai.com/docs/guides/gpt/chat-completions-api" target="_blank">{{__('More Info')}}<svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-arrow-up-right" width="24" height="24" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"></path><path d="M17 7l-10 10"></path><path d="M8 7l9 0l0 9"></path></svg></a>--}}
						{{--							</div>--}}
						{{--						</div>--}}
						<button @if($app_is_demo) type="button" onclick="return toastr.info('This feature is disabled in Demo version.')" @else type="submit" @endif class="btn btn-primary !py-3 w-100">
							{{__('Save')}}
						</button>
					</form>
				</div>
			</div>
		</div>
	</div>

@endsection

@section('script')
	<script src="{{ custom_theme_url('assets/js/panel/openai.js') }}"></script>
	<script src="{{ custom_theme_url('assets/select2/select2.min.js') }}"></script>
	<script src="{{ custom_theme_url('assets/libs/ace/src-min-noconflict/ace.js') }}" type="text/javascript" charset="utf-8"></script>
	<style type="text/css" media="screen">
		#chat_completions {
			position: absolute;
			top: 0;
			right: 0;
			bottom: 0;
			left: 0;
		}
		.ace_editor{
			min-height: 200px;
		}
	</style>
	<script>
		var editor_chat_completions = ace.edit("chat_completions");
		//editor.setTheme("ace/theme/monokai");
		editor_chat_completions.session.setMode("ace/mode/json");
	</script>
@endsection

@push('script')
	<script>
		$(document).ready(function() {
			$('#chatbot_id').on('change', function () {

				let value = $(this).val();

				if(value == '0')
				{
					$('#chatbot_training_json').removeClass('d-none');
				}else{
					$('#chatbot_training_json').addClass('d-none');
				}
			});
		});
	</script>
@endpush
