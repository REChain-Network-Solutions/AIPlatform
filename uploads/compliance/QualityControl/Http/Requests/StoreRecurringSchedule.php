<?php

namespace Modules\QualityControl\Http\Requests;

use App\Http\Requests\CoreRequest;
use Carbon\Carbon;

class StoreRecurringSchedule extends CoreRequest
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
            'billing_cycle' => 'required'
        ];

        if (!$this->has('immediate_schedule')) {
            $rules['issue_date'] = 'required|date_format:"' . $setting->date_format . '"|after:'.Carbon::now()->format($setting->date_format);
        }



        return $rules;
    }

}
