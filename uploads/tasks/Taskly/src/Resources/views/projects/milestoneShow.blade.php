 <div class="modal-body">
     @if ($currentWorkspace && $milestone)
         <div class="p-2">
             <div class="row mb-4">
                 <div class="col-md-4">
                     <div>
                         <div class="form-control-label">{{ __('Milestone Title') }}</div>
                         <p class="mt-1">{{ $milestone->title }}</p>
                     </div>
                 </div>
                 <div class="col-md-4">
                     <div class="form-control-label">{{ __('Status') }}</div>
                     <p class="mt-1">
                         @if ($milestone->status == 'incomplete')
                             <label class="badge bg-warning p-2 px-3">{{ __('Incomplete') }}</label>
                         @endif
                         @if ($milestone->status == 'complete')
                             <label class="badge bg-success p-2 px-3">{{ __('Complete') }}</label>
                         @endif
                     </p>
                 </div>
                 <div class="col-md-4">
                     <div class="form-control-label">{{ __('Milestone Cost') }}</div>
                     <p class="mt-1">
                         {{ !empty($currentWorkspace->currency) ? $currentWorkspace->currency : '$' }}{{ number_format($milestone->cost) }}
                     </p>
                 </div>
                 <div class="col-md-12">
                     <div class="form-control-label">{{ __('Milestone Summary') }}</div>
                     <p class="mt-1">{{ $milestone->summary }}</p>
                 </div>
             </div>
         </div>
     @else
         <div class="container mt-5">
             <div class="card">
                 <div class="card-body p-4">
                     <div class="page-error">
                         <div class="page-inner">
                             <h1>404</h1>
                             <div class="page-description">
                                 {{ __('Page Not Found') }}
                             </div>
                             <div class="page-search">
                                 <p class="text-muted mt-3">
                                     {{ __("It's looking like you may have taken a wrong turn. Don't worry... it happens to the best of us. Here's a little tip that might help you get back on track.") }}
                                 </p>
                                 <div class="mt-3">
                                     <a class="btn-return-home badge-blue" href="{{ route('home') }}"><i
                                             class="fas fa-reply"></i> {{ __('Return Home') }}</a>
                                 </div>
                             </div>
                         </div>
                     </div>
                 </div>
             </div>
         </div>
     @endif
 </div>
