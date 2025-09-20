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
use Exception;
use ReturnTypeWillChange;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель Leg.
 *
 * Маршрут авиа перелета
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
class Leg extends AbstractObject implements LegInterface
{
    /**
     * Формат даты.
     */
    public const ISO8601 = 'Y-m-d';
    /**
     * Код аэропорта вылета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например LED.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 3)]
    #[Assert\Length(min: 3)]
    #[Assert\Regex("/[A-Z]{3}/")]
    private ?string $_departure_airport = null;

    /**
     * Код аэропорта прилета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например AMS.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 3)]
    #[Assert\Length(min: 3)]
    #[Assert\Regex("/[A-Z]{3}/")]
    private ?string $_destination_airport = null;

    /**
     * Дата вылета в формате YYYY-MM-DD по стандарту [ISO 8601:2004](http://www.iso.org/iso/catalogue_detail?csnumber=40874).
     *
     * @var DateTime|null
     */
    #[Assert\NotBlank]
    #[Assert\DateTime(format: self::ISO8601)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_departure_date = null;

    /**
     * Код авиакомпании по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx).
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 3)]
    #[Assert\Length(min: 2)]
    private ?string $_carrier_code = null;

    /**
     * Возвращает departure_airport.
     *
     * @return string|null
     */
    public function getDepartureAirport(): ?string
    {
        return $this->_departure_airport;
    }

    /**
     * Устанавливает departure_airport.
     *
     * @param string|null $value Код аэропорта вылета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например LED.
     *
     * @throws Exception
     * @return self
     */
    public function setDepartureAirport(?string $value = null): self
    {
        $this->_departure_airport = $this->validatePropertyValue('_departure_airport', $value);
        return $this;
    }

    /**
     * Возвращает destination_airport.
     *
     * @return string|null
     */
    public function getDestinationAirport(): ?string
    {
        return $this->_destination_airport;
    }

    /**
     * Устанавливает destination_airport.
     *
     * @param string|null $value Код аэропорта прилета по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx), например AMS.
     *
     * @throws Exception
     * @return self
     */
    public function setDestinationAirport(?string $value = null): self
    {
        $this->_destination_airport = $this->validatePropertyValue('_destination_airport', $value);
        return $this;
    }

    /**
     * Возвращает departure_date.
     *
     * @return DateTime|null
     */
    public function getDepartureDate(): ?DateTime
    {
        return $this->_departure_date;
    }

    /**
     * Устанавливает departure_date.
     *
     * @param DateTime|string|null $value Дата вылета в формате YYYY-MM-DD по стандарту [ISO 8601:2004](http://www.iso.org/iso/catalogue_detail?csnumber=40874).
     *
     * @throws Exception
     * @return self
     */
    public function setDepartureDate(mixed $value = null): self
    {
        $this->_departure_date = $this->validatePropertyValue('_departure_date', $value);
        return $this;
    }

    /**
     * Возвращает carrier_code.
     *
     * @return string|null
     */
    public function getCarrierCode(): ?string
    {
        return $this->_carrier_code;
    }

    /**
     * Устанавливает carrier_code.
     *
     * @param string|null $value Код авиакомпании по справочнику [IATA](https://www.iata.org/publications/Pages/code-search.aspx).
     *
     * @throws Exception
     * @return self
     */
    public function setCarrierCode(?string $value = null): self
    {
        $this->_carrier_code = $this->validatePropertyValue('_carrier_code', $value);
        return $this;
    }

    #[ReturnTypeWillChange]
    /**
     * Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.
     *
     * @return array Ассоциативный массив со свойствами текущего объекта
     */
    public function jsonSerialize(): array
    {
        $result = parent::jsonSerialize();
        $result['departure_date'] = $this->getDepartureDate()?->format(self::ISO8601);
        return $result;
    }
}
