<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace Tests\YooKassa\Request\SelfEmployed;

use Exception;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\SelfEmployed\SelfEmployedConfirmationType;
use YooKassa\Request\SelfEmployed\SelfEmployedRequest;
use YooKassa\Request\SelfEmployed\SelfEmployedRequestSerializer;

/**
 * SelfEmployedRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SelfEmployedRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new SelfEmployedRequestSerializer();
        $instance = SelfEmployedRequest::builder()->build($options);
        $data = $serializer->serialize($instance);

        $expected = [
            'itn' => $options['itn'],
            'phone' => $options['phone'],
        ];

        if (!empty($options['confirmation'])) {
            $expected['confirmation'] = $options['confirmation'];
        }

        self::assertEquals($expected, $data);
    }

    /**
     * @throws Exception
     */
    public static function validDataProvider(): array
    {
        $result = [
            [
                [
                    'itn' => Random::str(12, '0123456789'),
                    'phone' => Random::str(4, 15, '0123456789'),
                    'confirmation' => ['type' => Random::value(SelfEmployedConfirmationType::getEnabledValues())],
                ],
            ],
            [
                [
                    'itn' => Random::str(12, '0123456789'),
                    'phone' => Random::str(4, 15, '0123456789'),
                    'confirmation' => ['type' => Random::value(SelfEmployedConfirmationType::getEnabledValues())],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $request = [
                'itn' => Random::str(12, '0123456789'),
                'phone' => Random::str(4, 15, '0123456789'),
                'confirmation' => ['type' => Random::value(SelfEmployedConfirmationType::getEnabledValues())],
            ];
            $result[] = [$request];
        }

        return $result;
    }
}
