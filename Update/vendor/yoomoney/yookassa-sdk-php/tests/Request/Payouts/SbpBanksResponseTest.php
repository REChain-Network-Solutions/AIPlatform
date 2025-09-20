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

namespace Tests\YooKassa\Request\Payouts;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Deal\SafeDeal;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payouts\SbpBanksResponse;

/**
 * SbpBanksResponseTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class SbpBanksResponseTest extends AbstractTestCase
{
    protected SbpBanksResponse $object;

    /**
     * @param mixed|null $value
     * @return SbpBanksResponse
     */
    protected function getTestInstance(mixed $value = null): SbpBanksResponse
    {
        return new SbpBanksResponse($value);
    }

    /**
     * @return void
     */
    public function testSbpBanksResponseClassExists(): void
    {
        $this->object = $this->getMockBuilder(SbpBanksResponse::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SbpBanksResponse::class));
        $this->assertInstanceOf(SbpBanksResponse::class, $this->object);
    }

    /**
     * Test property "items"
     * @dataProvider validClassDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testItems(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertIsObject($instance->getItems());
        self::assertIsObject($instance->items);
        self::assertNotNull($instance->getItems());
        self::assertNotNull($instance->items);
        $instance = $this->getTestInstance($value);
        self::assertIsObject($instance->getItems());
        self::assertIsObject($instance->items);
        self::assertNotNull($instance->getItems());
        self::assertNotNull($instance->items);
        foreach ($value['items'] as $key => $element) {
            if (is_array($element) && !empty($element)) {
                self::assertEquals($element, $instance->getItems()[$key]->toArray());
                self::assertEquals($element, $instance->items[$key]->toArray());
                self::assertIsArray($instance->getItems()[$key]->toArray());
                self::assertIsArray($instance->items[$key]->toArray());
            }
            if (is_object($element) && !empty($element)) {
                self::assertEquals($element, $instance->getItems()->get($key));
                self::assertIsObject($instance->getItems()->get($key));
                self::assertIsObject($instance->items->get($key));
                self::assertIsObject($instance->getItems());
                self::assertIsObject($instance->items);
            }
        }
        self::assertCount(count($value['items']), $instance->getItems());
        self::assertCount(count($value['items']), $instance->items);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validItemsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_items'));
    }

    /**
     * @return array
     * @throws Exception
     */
    public function validClassDataProvider(): array
    {
        $result = [];
        for ($i = 0; $i < 4; $i++) {
            $result[] = $this->getValidDataProviderByClass(new SbpBanksResponse());
        }
        return $result;
    }
}
