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

namespace Tests\YooKassa\Model\SavePaymentMethod;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use YooKassa\Model\SavePaymentMethod\SavePaymentMethod;

/**
 * SavePaymentMethodTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
*/
class SavePaymentMethodTest extends AbstractTestCase
{
    protected SavePaymentMethod $object;

    /**
    * @return SavePaymentMethod
    */
    protected function getTestInstance(): SavePaymentMethod
    {
        return new SavePaymentMethod();
    }

    /**
    * @return void
    */
    public function testPaymentMethodClassExists(): void
    {
        $this->object = $this->getMockBuilder(SavePaymentMethod::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(SavePaymentMethod::class));
        $this->assertInstanceOf(SavePaymentMethod::class, $this->object);
    }

    /**
    * Test property "type"
    * @dataProvider validTypeDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testType(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setType($value);
        self::assertNotNull($instance->getType());
        self::assertNotNull($instance->type);
        self::assertEquals($value, is_array($value) ? $instance->getType()->toArray() : $instance->getType());
        self::assertEquals($value, is_array($value) ? $instance->type->toArray() : $instance->type);
    }

    /**
    * Test invalid property "type"
    * @dataProvider invalidTypeDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidType(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setType($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidTypeDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_type'));
    }

    /**
    * Test property "id"
    * @dataProvider validIdDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testId(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setId($value);
        self::assertNotNull($instance->getId());
        self::assertNotNull($instance->id);
        self::assertEquals($value, is_array($value) ? $instance->getId()->toArray() : $instance->getId());
        self::assertEquals($value, is_array($value) ? $instance->id->toArray() : $instance->id);
    }

    /**
    * Test invalid property "id"
    * @dataProvider invalidIdDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidId(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setId($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidIdDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_id'));
    }

    /**
    * Test property "saved"
    * @dataProvider validSavedDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testSaved(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setSaved($value);
        self::assertNotNull($instance->getSaved());
        self::assertNotNull($instance->saved);
        self::assertEquals($value, is_array($value) ? $instance->getSaved()->toArray() : $instance->getSaved());
        self::assertEquals($value, is_array($value) ? $instance->saved->toArray() : $instance->saved);
        self::assertIsBool($instance->getSaved());
        self::assertIsBool($instance->saved);
    }

    /**
    * Test invalid property "saved"
    * @dataProvider invalidSavedDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidSaved(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setSaved($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validSavedDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_saved'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidSavedDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_saved'));
    }

    /**
    * Test property "status"
    * @dataProvider validStatusDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testStatus(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setStatus($value);
        self::assertNotNull($instance->getStatus());
        self::assertNotNull($instance->status);
        self::assertEquals($value, is_array($value) ? $instance->getStatus()->toArray() : $instance->getStatus());
        self::assertEquals($value, is_array($value) ? $instance->status->toArray() : $instance->status);
    }

    /**
    * Test invalid property "status"
    * @dataProvider invalidStatusDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidStatus(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setStatus($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidStatusDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_status'));
    }

    /**
    * Test property "holder"
    * @dataProvider validHolderDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testHolder(mixed $value): void
    {
        $instance = $this->getTestInstance();
        $instance->setHolder($value);
        self::assertNotNull($instance->getHolder());
        self::assertNotNull($instance->holder);
        self::assertEquals($value, is_array($value) ? $instance->getHolder()->toArray() : $instance->getHolder());
        self::assertEquals($value, is_array($value) ? $instance->holder->toArray() : $instance->holder);
    }

    /**
    * Test invalid property "holder"
    * @dataProvider invalidHolderDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidHolder(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setHolder($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validHolderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_holder'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidHolderDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_holder'));
    }

    /**
    * Test property "title"
    * @dataProvider validTitleDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testTitle(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getTitle());
        self::assertEmpty($instance->title);
        $instance->setTitle($value);
        self::assertEquals($value, is_array($value) ? $instance->getTitle()->toArray() : $instance->getTitle());
        self::assertEquals($value, is_array($value) ? $instance->title->toArray() : $instance->title);
        if (!empty($value)) {
            self::assertNotNull($instance->getTitle());
            self::assertNotNull($instance->title);
        }
    }

    /**
    * Test invalid property "title"
    * @dataProvider invalidTitleDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidTitle(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTitle($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validTitleDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_title'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidTitleDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_title'));
    }

    /**
    * Test property "confirmation"
    * @dataProvider validConfirmationDataProvider
    * @param mixed $value
    *
    * @return void
    * @throws Exception
    */
    public function testConfirmation(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getConfirmation());
        self::assertEmpty($instance->confirmation);
        $instance->setConfirmation($value);
        self::assertEquals($value, is_array($value) ? $instance->getConfirmation()->toArray() : $instance->getConfirmation());
        self::assertEquals($value, is_array($value) ? $instance->confirmation->toArray() : $instance->confirmation);
        if (!empty($value)) {
            self::assertNotNull($instance->getConfirmation());
            self::assertNotNull($instance->confirmation);
        }
    }

    /**
    * Test invalid property "confirmation"
    * @dataProvider invalidConfirmationDataProvider
    * @param mixed $value
    * @param string $exceptionClass
    *
    * @return void
    */
    public function testInvalidConfirmation(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setConfirmation($value);
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function validConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
    * @return array[]
    * @throws Exception
    */
    public function invalidConfirmationDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_confirmation'));
    }

    /**
     * Test property "card"
     * @dataProvider validCardDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testCard(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getCard());
        self::assertEmpty($instance->card);
        $instance->setCard($value);
        self::assertEquals($value, is_array($value) ? $instance->getCard()->toArray() : $instance->getCard());
        self::assertEquals($value, is_array($value) ? $instance->card->toArray() : $instance->card);
        if (!empty($value)) {
            self::assertNotNull($instance->getCard());
            self::assertNotNull($instance->card);
        }
    }

    /**
     * Test invalid property "card"
     * @dataProvider invalidCardDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidCard(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setCard($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validCardDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_card'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidCardDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_card'));
    }
}
