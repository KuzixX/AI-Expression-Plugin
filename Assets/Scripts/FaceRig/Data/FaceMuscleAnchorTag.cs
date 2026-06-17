namespace FaceRig.Data
{
    public enum FaceMuscleAnchorTag
    {
        //Тянут уголок губы к скуле (радость).
        lZygomaticusMajorLandmark,
        rZygomaticusMajorLandmark,

        // Поднимают верхнюю губу (Тянет к носу).
        lZygomaticusMinorLandmark,
        rZygomaticusMinorLandmark,

        // Опускает уголок губы (грусть).
        lDepressorAnguliOrisLandmark,
        rDepressorAnguliOrisLandmark,

        // Растягивают уголки губ в стороны.
        lRisoriusLandmark,
        rRisoriusLandmark,

        // Тянет внутренние части бровей вверх.
        lFrontalisInsideLandmark,
        rFrontalisInsideLandmark,

        // Тянет внешнюю части бровей вверх.
        lFrontalisOuterLandmark,
        rFrontalisOuterLandmark,

        // Опускают внутренню часть бровей вниз к носу.
        LCorrugatorSuperciliiLandmark,
        RCorrugatorSuperciliiLandmark,

        // Нужен для открытия челюсти.
        BridgeOfTheNose,

        // Поднимают верхнюю губу к носу.
        LLevatorLabiiSuperiorisLandmark,
        RLevatorLabiiSuperiorisLandmark,

        // Опускает нижнюю губу к подбородку
        LDepressorLabiiInferiorisLandmark,
        RDepressorLabiiInferiorisLandmark,
        
        // Мышца которая идет вокруг рта и делает визему О. По суути - это центр рта.
        OrbicularisOrisCenter,
        
        LOrbicularisOculi,
        ROrbicularisOculi,

        // Горизонтальный сдвиг челюсти (вычисляется из лэндмарков).
        JawSlide,
    }
}
