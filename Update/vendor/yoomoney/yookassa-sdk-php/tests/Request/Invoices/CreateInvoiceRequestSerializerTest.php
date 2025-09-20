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

namespace Tests\YooKassa\Request\Invoices;

use DateTime;
use Exception;
use Faker\Provider\Internet;
use PHPUnit\Framework\TestCase;
use YooKassa\Helpers\Random;
use YooKassa\Model\Invoice\LineItem;
use YooKassa\Model\Metadata;
use YooKassa\Model\MonetaryAmount;
use YooKassa\Request\Invoices\CreateInvoiceRequest;
use YooKassa\Request\Invoices\CreateInvoiceRequestSerializer;
use YooKassa\Request\Invoices\PaymentData;

/**
 * CreateInvoiceRequestSerializerTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class CreateInvoiceRequestSerializerTest extends TestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @param mixed $options
     */
    public function testSerialize(mixed $options): void
    {
        $serializer = new CreateInvoiceRequestSerializer();
        $instance = CreateInvoiceRequest::builder()->build($options);
        $data = $serializer->serialize($instance);

        $request = new CreateInvoiceRequest($options);
        $expected = $request->toArray();

        self::assertEquals($expected, $data);
    }

    /**
     * @throws Exception
     */
    public static function validDataProvider(): array
    {
        $metadata = new Metadata();
        $metadata->test = 'test';
        $result = [
            [
                [
                    'payment_data' => [
                        'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                    ],
                    'cart' => [
                        [
                            'description' => Random::str(5, 128),
                            'price' => new MonetaryAmount(Random::int(1, 1000000)),
                            'discount_price' => null,
                            'quantity' => Random::int(1, 10),
                        ]
                    ],
                    'delivery_method_data' => null,
                    'expires_at' => new DateTime('+ 1 hour'),
                    'locale' => null,
                    'description' => null,
                    'metadata' => null,
                ],
            ],
            [
                [
                    'payment_data' => [
                        'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                        'save_payment_method' => true,
                        'description' => Random::str(5, 128),
                        'client_ip' => Internet::localIpv4(),
                    ],
                    'cart' => [
                        new LineItem([
                            'description' => Random::str(5, 128),
                            'price' => new MonetaryAmount(Random::int(1, 1000000)),
                            'discount_price' => null,
                            'quantity' => Random::int(1, 10),
                        ]),
                    ],
                    'delivery_method_data' => ['type' => 'self'],
                    'expires_at' => new DateTime('+ 1 day'),
                    'locale' => Random::value(['ru_RU', 'en_US', null]),
                    'description' => '',
                    'metadata' => [Random::str(1)],
                ],
            ],
        ];
        for ($i = 0; $i < 10; $i++) {
            $even = ($i % 3);
            $request = [
                'payment_data' => new PaymentData([
                    'amount' => new MonetaryAmount(Random::int(1, 1000000)),
                    'save_payment_method' => true,
                    'description' => Random::str(5, 128),
                    'client_ip' => Internet::localIpv4(),
                ]),
                'cart' => [
                    $even ? [
                        'description' => Random::str(5, 128),
                        'price' => new MonetaryAmount(Random::int(1, 1000000)),
                        'discount_price' => null,
                        'quantity' => Random::int(1, 10),
                    ] : new LineItem([
                        'description' => Random::str(5, 128),
                        'price' => new MonetaryAmount(Random::int(1, 1000000)),
                        'discount_price' => null,
                        'quantity' => Random::int(1, 10),
                    ]),
                ],
                'delivery_method_data' => $even ? ['type' => 'self'] : null,
                'expires_at' => $even ? new DateTime('+ 1 month') : '2024-10-18T10:51:18.139Z',
                'locale' => Random::value(['ru_RU', 'en_US', null]),
                'description' => Random::str(5, 128),
                'metadata' => $even ? $metadata : ['test' => 'test'],
            ];
            $result[] = [$request];
        }

        return $result;
    }


}
