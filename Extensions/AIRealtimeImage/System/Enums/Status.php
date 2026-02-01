<?php

namespace App\Extensions\AIRealtimeImage\System\Enums;

enum Status: string
{
    case pending = 'pending';

    case failed = 'failed';

    case success = 'success';
}
