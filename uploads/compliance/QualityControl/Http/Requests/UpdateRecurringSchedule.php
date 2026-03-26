<?php

namespace Modules\QualityControl\Http\Requests;

use App\Http\Requests\CoreRequest;
use Carbon\Carbon;

class UpdateRecurringSchedule extends CoreRequest
{

    /**
     * Determine if the user is authorized to make this request.
     *
     * @return bool
     */
    public function authorize()
    {
        return true;
    }

    /**
     * Get the validation rules that apply to the request.
     *
     * @return array
     */
    public function rules()
    {


        $setting = company();

        $rules = [

        ];

        if ($this->schedule_count == 0) {
            $rules['issue_date'] = 'required|date_format:"' . $setting->date_format . '"|after_or_equal:'.Carbon::now()->format($setting->date_format);


        }





        return $rules;
    }

}
