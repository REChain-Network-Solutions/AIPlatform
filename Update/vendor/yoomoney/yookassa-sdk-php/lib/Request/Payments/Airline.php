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

namespace YooKassa\Request\Payments;

use YooKassa\Common\AbstractObject;
use YooKassa\Common\ListObject;
use YooKassa\Common\ListObjectInterface;
use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Exceptions\InvalidPropertyValueTypeException;

/**
 * Класс, представляющий модель Airline.
 *
 * Объект с данными для [продажи авиабилетов](/developers/payment-acceptance/scenario-extensions/airline-tickets).
 * Используется только для платежей банковской картой.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $bookingReference Номер бронирования. Обязателен на этапе создания платежа
 * @property string $booking_reference Номер бронирования. Обязателен на этапе создания платежа
 * @property string $ticketNumber Уникальный номер билета. Обязателен на этапе подтверждения платежа
 * @property string $ticket_number Уникальный номер билета. Обязателен на этапе подтверждения платежа
 * @property ListObjectInterface|PassengerInterface[] $passengers Список пассажиров
 * @property ListObjectInterface|LegInterface[] $legs Список маршрутов
 */
class Airline extends AbstractObject implements AirlineInterface
{
    /**
     * Уникальный номер билета. Если при создании платежа вы уже знаете номер билета, тогда `ticket_number` — обязательный параметр. Если не знаете, тогда вместо `ticket_number` необходимо передать `booking_reference` с номером бронирования.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 150)]
    #[Assert\Length(min: 1)]
    #[Assert\Regex("/^[0-9]{1,150}$/")]
    private ?string $_ticket_number = null;

    /**
     * Номер бронирования. Обязателен, если не передан `ticket_number`.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 20)]
    #[Assert\Length(min: 1)]
    private ?string $_booking_reference = null;

    /**
     * Список пассажиров.
     *
     * @var PassengerInterface[]|ListObjectInterface|null
     */
    #[Assert\Valid]
    #[Assert\AllType(Passenger::class)]
    #[Assert\Type(ListObject::class)]
    #[Assert\Count(max: 500)]
    #[Assert\Count(min: 0)]
    private ?ListObject $_passengers = null;

    /**
     * Список перелетов.
     *
     * @var LegInterface[]|ListObjectInterface|null
     */
    #[Assert\Valid]
    #[Assert\AllType(Leg::class)]
    #[Assert\Type(ListObject::class)]
    #[Assert\Count(max: 4)]
    #[Assert\Count(min: 0)]
    private ?ListObject $_legs = null;

    /**
     * Возвращает ticket_number.
     *
     * @return string|null
     */
    public function getTicketNumber(): ?string
    {
        return $this->_ticket_number;
    }

    /**
     * Устанавливает ticket_number.
     *
     * @param string|null $ticket_number Уникальный номер билета. Если при создании платежа вы уже знаете номер билета, тогда `ticket_number` — обязательный параметр. Если не знаете, тогда вместо `ticket_number` необходимо передать `booking_reference` с номером бронирования.
     *
     * @return self
     */
    public function setTicketNumber(?string $ticket_number = null): self
    {
        $this->_ticket_number = $this->validatePropertyValue('_ticket_number', $ticket_number);
        return $this;
    }

    /**
     * Возвращает booking_reference.
     *
     * @return string|null
     */
    public function getBookingReference(): ?string
    {
        return $this->_booking_reference;
    }

    /**
     * Устанавливает booking_reference.
     *
     * @param string|null $booking_reference Номер бронирования. Обязателен, если не передан `ticket_number`.
     *
     * @return self
     */
    public function setBookingReference(?string $booking_reference = null): self
    {
        $this->_booking_reference = $this->validatePropertyValue('_booking_reference', $booking_reference);
        return $this;
    }

    /**
     * Возвращает список пассажиров.
     *
     * @return PassengerInterface[]|ListObjectInterface Список пассажиров.
     */
    public function getPassengers(): ListObjectInterface
    {
        if ($this->_passengers === null) {
            $this->_passengers = new ListObject(Passenger::class);
        }
        return $this->_passengers;
    }

    /**
     * Устанавливает список пассажиров.
     *
     * @param ListObjectInterface|array|null $passengers Список пассажиров.
     *
     * @return self
     */
    public function setPassengers(mixed $passengers = null): self
    {
        $this->_passengers = $this->validatePropertyValue('_passengers', $passengers);
        return $this;
    }

    /**
     * Добавляет пассажира.
     *
     * @param array|PassengerInterface $value Пассажир
     *
     * @return self
     */
    public function addPassenger(mixed $value): self
    {
        $this->getPassengers()->add($value);
        return $this;
    }

    /**
     * Возвращает legs.
     *
     * @return LegInterface[]|ListObjectInterface
     */
    public function getLegs(): ListObjectInterface
    {
        if ($this->_legs === null) {
            $this->_legs = new ListObject(Leg::class);
        }
        return $this->_legs;
    }

    /**
     * Устанавливает legs.
     *
     * @param ListObjectInterface|array|null $legs Список перелетов.
     *
     * @return self
     */
    public function setLegs(mixed $legs = null): self
    {
        $this->_legs = $this->validatePropertyValue('_legs', $legs);
        return $this;
    }

    /**
     * Добавляет объект-контейнер с данными о маршруте.
     *
     * @param array|LegInterface $value Объект-контейнер с данными о маршруте
     *
     * @return self
     */
    public function addLeg(mixed $value): self
    {
        $this->getLegs()->add($value);
        return $this;
    }

    /**
     * Проверка на наличие данных.
     */
    public function notEmpty(): bool
    {
        return !$this->getLegs()->isEmpty() || !$this->getPassengers()->isEmpty() || $this->_ticket_number || $this->_booking_reference;
    }
}
