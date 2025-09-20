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

namespace Tests\YooKassa\Request\Payments;

use Exception;
use Tests\YooKassa\AbstractTestCase;
use Datetime;
use YooKassa\Model\Metadata;
use YooKassa\Request\Payments\Airline;

/**
 * AirlineTest
 *
 * @category    ClassTest
 * @author      cms@yoomoney.ru
 * @link        https://yookassa.ru/developers/api
 */
class AirlineTest extends AbstractTestCase
{
    protected Airline $object;

    /**
     * @return Airline
     */
    protected function getTestInstance(): Airline
    {
        return new Airline();
    }

    /**
     * @return void
     */
    public function testAirlineClassExists(): void
    {
        $this->object = $this->getMockBuilder(Airline::class)->getMockForAbstractClass();
        $this->assertTrue(class_exists(Airline::class));
        $this->assertInstanceOf(Airline::class, $this->object);
    }

    /**
     * Test property "ticket_number"
     * @dataProvider validTicketNumberDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testTicketNumber(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getTicketNumber());
        self::assertEmpty($instance->ticket_number);
        $instance->setTicketNumber($value);
        self::assertEquals($value, is_array($value) ? $instance->getTicketNumber()->toArray() : $instance->getTicketNumber());
        self::assertEquals($value, is_array($value) ? $instance->ticket_number->toArray() : $instance->ticket_number);
        if (!empty($value)) {
            self::assertNotNull($instance->getTicketNumber());
            self::assertNotNull($instance->ticket_number);
            self::assertMatchesRegularExpression("/[0-9]{1,150}/", $instance->getTicketNumber());
            self::assertMatchesRegularExpression("/[0-9]{1,150}/", $instance->ticket_number);
            self::assertLessThanOrEqual(150, is_string($instance->getTicketNumber()) ? mb_strlen($instance->getTicketNumber()) : $instance->getTicketNumber());
            self::assertLessThanOrEqual(150, is_string($instance->ticket_number) ? mb_strlen($instance->ticket_number) : $instance->ticket_number);
            self::assertGreaterThanOrEqual(1, is_string($instance->getTicketNumber()) ? mb_strlen($instance->getTicketNumber()) : $instance->getTicketNumber());
            self::assertGreaterThanOrEqual(1, is_string($instance->ticket_number) ? mb_strlen($instance->ticket_number) : $instance->ticket_number);
        }
    }

    /**
     * Test invalid property "ticket_number"
     * @dataProvider invalidTicketNumberDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidTicketNumber(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setTicketNumber($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validTicketNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_ticket_number'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidTicketNumberDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_ticket_number'));
    }

    /**
     * Test property "booking_reference"
     * @dataProvider validBookingReferenceDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testBookingReference(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getBookingReference());
        self::assertEmpty($instance->booking_reference);
        $instance->setBookingReference($value);
        self::assertEquals($value, is_array($value) ? $instance->getBookingReference()->toArray() : $instance->getBookingReference());
        self::assertEquals($value, is_array($value) ? $instance->booking_reference->toArray() : $instance->booking_reference);
        if (!empty($value)) {
            self::assertNotNull($instance->getBookingReference());
            self::assertNotNull($instance->booking_reference);
            self::assertLessThanOrEqual(20, is_string($instance->getBookingReference()) ? mb_strlen($instance->getBookingReference()) : $instance->getBookingReference());
            self::assertLessThanOrEqual(20, is_string($instance->booking_reference) ? mb_strlen($instance->booking_reference) : $instance->booking_reference);
            self::assertGreaterThanOrEqual(1, is_string($instance->getBookingReference()) ? mb_strlen($instance->getBookingReference()) : $instance->getBookingReference());
            self::assertGreaterThanOrEqual(1, is_string($instance->booking_reference) ? mb_strlen($instance->booking_reference) : $instance->booking_reference);
        }
    }

    /**
     * Test invalid property "booking_reference"
     * @dataProvider invalidBookingReferenceDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidBookingReference(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setBookingReference($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validBookingReferenceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_booking_reference'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidBookingReferenceDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_booking_reference'));
    }

    /**
     * Test property "passengers"
     * @dataProvider validPassengersDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testPassengers(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getPassengers());
        self::assertEmpty($instance->passengers);
        self::assertIsObject($instance->getPassengers());
        self::assertIsObject($instance->passengers);
        self::assertCount(0, $instance->getPassengers());
        self::assertCount(0, $instance->passengers);
        if (!empty($value)) {
            self::assertNotNull($instance->getPassengers());
            self::assertNotNull($instance->passengers);
            foreach ($value as $element) {
                if (is_object($element) && !empty($element)) {
                    $instance->addPassenger($element);
                    self::assertEquals($element, $instance->getPassengers()->get(0));
                    self::assertIsObject($instance->getPassengers()->get(0));
                    self::assertIsObject($instance->passengers->get(0));
                    self::assertIsObject($instance->getPassengers());
                    self::assertIsObject($instance->passengers);
                    $instance->getPassengers()->clear();
                }
            }

            $instance->setPassengers($value);
            self::assertTrue($instance->notEmpty());
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getPassengers()[$key]->toArray());
                    self::assertEquals($element, $instance->passengers[$key]->toArray());
                    self::assertIsArray($instance->getPassengers()[$key]->toArray());
                    self::assertIsArray($instance->passengers[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getPassengers()->get($key));
                    self::assertIsObject($instance->getPassengers()->get($key));
                    self::assertIsObject($instance->passengers->get($key));
                    self::assertIsObject($instance->getPassengers());
                    self::assertIsObject($instance->passengers);
                }
            }
            self::assertCount(count($value), $instance->getPassengers());
            self::assertCount(count($value), $instance->passengers);
            self::assertLessThanOrEqual(500, $instance->getPassengers()->count());
            self::assertLessThanOrEqual(500, $instance->passengers->count());
            self::assertGreaterThanOrEqual(0, $instance->getPassengers()->count());
            self::assertGreaterThanOrEqual(0, $instance->passengers->count());
        }
    }

    /**
     * Test invalid property "passengers"
     * @dataProvider invalidPassengersDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidPassengers(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setPassengers($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validPassengersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_passengers'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidPassengersDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_passengers'));
    }

    /**
     * Test property "legs"
     * @dataProvider validLegsDataProvider
     * @param mixed $value
     *
     * @return void
     * @throws Exception
     */
    public function testLegs(mixed $value): void
    {
        $instance = $this->getTestInstance();
        self::assertEmpty($instance->getLegs());
        self::assertEmpty($instance->legs);
        self::assertIsObject($instance->getLegs());
        self::assertIsObject($instance->legs);
        self::assertCount(0, $instance->getLegs());
        self::assertCount(0, $instance->legs);
        if (!empty($value)) {
            self::assertNotNull($instance->getLegs());
            self::assertNotNull($instance->legs);
            foreach ($value as $element) {
                if (is_object($element) && !empty($element)) {
                    $instance->addLeg($element);
                    self::assertEquals($element, $instance->getLegs()->get(0));
                    self::assertIsObject($instance->getLegs()->get(0));
                    self::assertIsObject($instance->legs->get(0));
                    self::assertIsObject($instance->getLegs());
                    self::assertIsObject($instance->legs);
                    $instance->getLegs()->clear();
                }
            }

            $instance->setLegs($value);
            self::assertTrue($instance->notEmpty());
            foreach ($value as $key => $element) {
                if (is_array($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getLegs()[$key]->toArray());
                    self::assertEquals($element, $instance->legs[$key]->toArray());
                    self::assertIsArray($instance->getLegs()[$key]->toArray());
                    self::assertIsArray($instance->legs[$key]->toArray());
                }
                if (is_object($element) && !empty($element)) {
                    self::assertEquals($element, $instance->getLegs()->get($key));
                    self::assertIsObject($instance->getLegs()->get($key));
                    self::assertIsObject($instance->legs->get($key));
                    self::assertIsObject($instance->getLegs());
                    self::assertIsObject($instance->legs);
                }
            }
            self::assertCount(count($value), $instance->getLegs());
            self::assertCount(count($value), $instance->legs);
            self::assertLessThanOrEqual(4, $instance->getLegs()->count());
            self::assertLessThanOrEqual(4, $instance->legs->count());
            self::assertGreaterThanOrEqual(0, $instance->getLegs()->count());
            self::assertGreaterThanOrEqual(0, $instance->legs->count());
        }
    }

    /**
     * Test invalid property "legs"
     * @dataProvider invalidLegsDataProvider
     * @param mixed $value
     * @param string $exceptionClass
     *
     * @return void
     */
    public function testInvalidLegs(mixed $value, string $exceptionClass): void
    {
        $instance = $this->getTestInstance();

        $this->expectException($exceptionClass);
        $instance->setLegs($value);
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function validLegsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getValidDataProviderByType($instance->getValidator()->getRulesByPropName('_legs'));
    }

    /**
     * @return array[]
     * @throws Exception
     */
    public function invalidLegsDataProvider(): array
    {
        $instance = $this->getTestInstance();
        return $this->getInvalidDataProviderByType($instance->getValidator()->getRulesByPropName('_legs'));
    }
}
