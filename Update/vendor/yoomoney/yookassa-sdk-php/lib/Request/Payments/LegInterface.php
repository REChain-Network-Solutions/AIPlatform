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

use DateTime;

/**
 * Interface PassengerInterface.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $departureAirport Трёхбуквенный IATA-код аэропорта вылета
 * @property string $departure_airport Трёхбуквенный IATA-код аэропорта вылета
 * @property string $destinationAirport Трёхбуквенный IATA-код аэропорта прилёта
 * @property string $destination_airport Трёхбуквенный IATA-код аэропорта прилёта
 * @property string $departureDate Дата вылета в формате YYYY-MM-DD ISO 8601:2004
 * @property string $departure_date Дата вылета в формате YYYY-MM-DD ISO 8601:2004
 * @property string $carrierCode Код авиакомпании по справочнику
 * @property string $carrier_code Код авиакомпании по справочнику
 */
interface LegInterface
{
    /**
     * Возвращает трёхбуквенный IATA-код аэропорта вылета.
     *
     * @return string|null Трёхбуквенный IATA-код аэропорта вылета
     */
    public function getDepartureAirport(): ?string;

    /**
     * Возвращает трёхбуквенный IATA-код аэропорта прилёта.
     *
     * @return string|null Трёхбуквенный IATA-код аэропорта прилёта
     */
    public function getDestinationAirport(): ?string;

    /**
     * Возвращает дату вылета в формате YYYY-MM-DD ISO 8601:2004.
     *
     * @return DateTime|null Дата вылета в формате YYYY-MM-DD ISO 8601:2004
     */
    public function getDepartureDate(): ?DateTime;

    /**
     * Устанавливает трёхбуквенный IATA-код аэропорта вылета.
     *
     * @param string|null $value Трёхбуквенный IATA-код аэропорта вылета
     *
     * @return self
     */
    public function setDepartureAirport(?string $value): self;

    /**
     * Устанавливает трёхбуквенный IATA-код аэропорта прилёта.
     *
     * @param string|null $value Трёхбуквенный IATA-код аэропорта прилёта
     *
     * @return self
     */
    public function setDestinationAirport(?string $value): self;

    /**
     * Устанавливает дату вылета в формате YYYY-MM-DD ISO 8601:2004.
     *
     * @param DateTime|string|null $value Дата вылета в формате YYYY-MM-DD ISO 8601:2004
     *
     * @return self
     */
    public function setDepartureDate(mixed $value): self;
}
