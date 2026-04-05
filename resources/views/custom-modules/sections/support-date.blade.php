@php $supportDate = \Carbon\Carbon::parse($fetchSetting->supported_until) @endphp

@if ($supportDate->isPast())
    <span>Your support has been expired on <b>{{ $supportDate->translatedFormat('d M, Y') }}</b>
        @if($supportDate->isYesterday())
            (Yesterday)
        @endif
    </span>
    <br>
@else
    <span >Your support will expire on <b>{{ $supportDate->translatedFormat('d M, Y') }}</b>
        @if($supportDate->isToday())
            (Today)
        @elseif($supportDate->isTomorrow())
            (Tomorrow)
        @endif
    </span>
    @if($supportDate->diffInDays() < 90)
        <div class="h-mt2 mt-2">
            <p class="t-body -size-m -color-mid">
                <a class="img-lightbox"
                   data-image-url="{{ asset('img/Support_Extension_Cost.jpg') }}"
                   href="javascript:;">How much do I save by extending now?
                </a>
            </p>
        </div>
    @endif
@endif



