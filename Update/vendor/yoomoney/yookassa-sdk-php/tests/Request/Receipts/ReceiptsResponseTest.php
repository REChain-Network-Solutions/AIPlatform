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

namespace Tests\YooKassa\Request\Receipts;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Common\ListObject;
use YooKassa\Helpers\Random;
use YooKassa\Helpers\UUID;
use YooKassa\Model\Receipt\ReceiptType;
use YooKassa\Model\Receipt\SettlementType;
use YooKassa\Request\Receipts\AbstractReceiptResponse;
use YooKassa\Request\Receipts\ReceiptResponseInterface;
use YooKassa\Request\Receipts\ReceiptsResponse;

/**
 * ReceiptsResponseTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class ReceiptsResponseTest extends AbstractTestCase
{
    /**
     * @dataProvider validDataProvider
     *
     * @throws Exception
     */
    public function testGetType(array $options): void
    {
        $instance = new ReceiptsResponse($options);

        self::assertEquals($options['type'], $instance->getType());
        self::assertEquals($options['next_cursor'], $instance->getNextCursor());
    }

    /**
     * @dataProvider validDataProvider
     *
     * @throws Exception
     */
    public function testGetItems(array $options): void
    {
        $instance = new ReceiptsResponse($options);

        self::assertEquals(count($options['items']), count($instance->getItems()));
        self::assertTrue($instance->hasNextCursor());

        foreach ($instance->getItems() as $index => $item) {
            self::assertInstanceOf(ReceiptResponseInterface::class, $item);
            self::assertArrayHasKey($index, $options['items']);
            self::assertEquals($options['items'][$index]['id'], $item->getId());
            self::assertEquals($options['items'][$index]['type'], $item->getType());
            self::assertEquals($options['items'][$index]['tax_system_code'], $item->getTaxSystemCode());
            self::assertEquals($options['items'][$index]['status'], $item->getStatus());

            self::assertEquals(count($options['items'][$index]['items']), count($item->getItems()));
            $instance->getItems()->clear();
            self::assertInstanceOf(ListObject::class, $instance->getItems());
        }
    }

    public function validDataProvider(): array
    {
        return [
            [
                [
                    'type' => 'list',
                    'items' => $this->generateReceipts(),
                    'next_cursor' => Random::str(36),
                ],
            ],
        ];
    }

    private function generateReceipts(): array
    {
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateReceipt();
        }

        return $return;
    }

    private function generateReceipt(): array
    {
        $type = Random::value(ReceiptType::getEnabledValues());

        return [
            'id' => Random::str(39),
            'type' => $type,
            'status' => Random::value(['pending', 'succeeded', 'canceled']),
            'items' => $this->generateItems(),
            'settlements' => $this->generateSettlements(),
            'tax_system_code' => Random::int(1, 6),
            $type . '_id' => UUID::v4(),
        ];
    }

    private function generateItems(): array
    {
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateItem();
        }

        return $return;
    }

    private function generateItem(): array
    {
        return [
            'description' => Random::str(1, 128),
            'amount' => [
                'value' => round(Random::float(1.00, 100.00), 2),
                'currency' => 'RUB',
            ],
            'quantity' => round(Random::float(0.001, 99.999), 3),
            'vat_code' => Random::int(1, 6),
        ];
    }

    private function generateSettlements(): array
    {
        $return = [];
        $count = Random::int(1, 10);

        for ($i = 0; $i < $count; $i++) {
            $return[] = $this->generateSettlement();
        }

        return $return;
    }

    private function generateSettlement(): array
    {
        return [
            'type' => Random::value(SettlementType::getValidValues()),
            'amount' => [
                'value' => round(Random::float(1.00, 100.00), 2),
                'currency' => 'RUB',
            ],
        ];
    }
}
