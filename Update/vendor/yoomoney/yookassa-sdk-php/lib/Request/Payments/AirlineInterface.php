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

use YooKassa\Common\ListObjectInterface;

/**
 * Interface Airline.
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
 * @property PassengerInterface[] $passengers Список пассажиров
 * @property Leg[] $legs Список маршрутов
 */
interface AirlineInterface
{
    /**
     * Номер бронирования. Обязателен на этапе создания платежа.
     */
    public function getBookingReference(): ?string;

    /**
     * Уникальный номер билета. Обязателен на этапе подтверждения платежа.
     */
    public function getTicketNumber(): ?string;

    /**
     * Список объектов-контейнеров с данными пассажиров.
     *
     * @return PassengerInterface[]|ListObjectInterface
     */
    public function getPassengers(): ListObjectInterface;

    /**
     * Список объектов-контейнеров с данными о маршруте.
     *
     * @return LegInterface[]|ListObjectInterface
     */
    public function getLegs(): ListObjectInterface;
}
