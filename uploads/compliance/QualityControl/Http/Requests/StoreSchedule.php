<?php

namespace Modules\QualityControl\Http\Requests;

use App\Http\Requests\CoreRequest;


class StoreSchedule extends CoreRequest
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


        $rules = [

            'issue_date' => 'required',

        ];




        return $rules;
    }





}
